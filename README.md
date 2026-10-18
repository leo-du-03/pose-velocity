# pose-velocity

A UR5e arm with a LEAP hand holds a randomly generated tool. It moves the tool so that the **head center passes through a goal point, with the head axis aligned to a goal direction, while moving at a goal velocity**.

Training is staged:

- **Stage A (default, `grip_mode = "weld"`):** the tool is welded to the palm at a random grasp, and the policy controls only the arm. This learns the strike on its own.
- **Stage B (`grip_mode = "physical"`):** the hand has to grasp the tool from the table and keep hold of it. This path works but is not tuned yet.

The rest of the stack:

- **Simulator:** MuJoCo, running on CPU in subprocess workers.
- **Policy:** asymmetric PPO in PyTorch, running on CPU or CUDA.

## Install

```bash
conda env create -f environment.yml        # creates env "pose-velocity" (Python 3.10)
conda activate pose-velocity
```

To get a smaller CPU-only PyTorch, run this **before** the step above, or afterwards to replace the default build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

To train on a GPU under WSL:
1. Install the NVIDIA Windows driver.
2. Run `wsl --shutdown`, then reopen WSL.
3. Check that `nvidia-smi` works inside WSL.
4. Install a CUDA build of torch from pytorch.org.

The robot models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). `robot_descriptions` clones them into `~/.cache/robot_descriptions` on first use, which needs `git` and internet access. If you work offline, clone Menagerie yourself and point to it:

```bash
export POSEVEL_UR5E_XML=/path/to/mujoco_menagerie/universal_robots_ur5e/ur5e.xml
export POSEVEL_LEAP_XML=/path/to/mujoco_menagerie/leap_hand/right_hand.xml
```

## Verify, in this order

Run every command from the project root.

1. **Scene builds; robot is discovered correctly**
   ```bash
   python scene.py
   ```
   Expected output:
   - `arm joints (6)`
   - `hand joints (16)`
   - 4 fingertips, with the thumb last
   - `action dim: 6` (weld; 22 with `grip_mode = "physical"`)
   - an obs dim of about 100 (about 170 for physical grip)

   The first run downloads Menagerie and builds the reach map, which takes a few seconds.

2. **Look at the scene**
   ```bash
   python scene.py --view
   ```
   - Check that the hand is mounted sensibly on the flange.
   - With weld grip, the tool should sit in the closed hand. With physical grip it lies on the table in front of the arm.
   - Goal markers: the **green** sphere is the goal point, the **red** arrow is the goal velocity (its length scales with speed), and the **blue** arrow is the goal head axis.

   If the hand is rotated or offset wrongly, tune `SceneCfg.hand_mount_euler` and `SceneCfg.hand_mount_pos` in `config.py`.

3. **Unit and integration tests**
   ```bash
   pytest -q tests
   ```
   The tests cover:
   - joint and fingertip discovery;
   - 50 random tools compile within the mass bounds and settle on the table;
   - the weld constraint compiles and holds the tool under random arm motion;
   - head-twist actions move the head at the commanded velocity;
   - weld goals are reachable (their stored arm configuration puts the head on the goal), and the region sampler's constraints hold;
   - goal-plane crossings count only near the goal;
   - the Gymnasium `check_env`;
   - 1500-step random rollouts stay finite, for both grip modes;
   - soft, regrip and respawn reset semantics;
   - a scripted pass-through of the goal triggers success;
   - the scripted oracle succeeds on most goals.

4. **Check that goals are feasible with the scripted oracle**
   ```bash
   python evaluate.py --policy oracle --episodes 100            # level 0
   python evaluate.py --policy oracle --episodes 100 --level 1  # tightest tolerances, fastest goals
   ```
   The oracle flies the head to the lead-in point, aligns the axis, then moves at `v_g` through the goal. It should reach about 90% success at both levels. If it falls well below that after you change the env, the env is broken, not the learner.

5. **Watch the env live with random actions**
   ```bash
   python evaluate.py --policy random --render human --realtime --episodes 3
   ```
   You can also write a video instead: `python evaluate.py --policy oracle --episodes 3 --video oracle.mp4`

6. **Training smoke test** (runs in under a minute on CPU)
   ```bash
   python train.py --device cpu --num-envs 4 --total-steps 20000
   ```
   - The output should show iterations with `sps`, return, success, pass and drop rates, KL, clip fraction and lr.
   - It saves `runs/<name>/ckpt_latest.pt`, and `runs/<name>/ckpt_best.pt` whenever the success rate over the
     last 200 episodes beats the previous best at the same or a higher curriculum level.
   - `--device auto` uses CUDA when available.

7. **Real training run and monitoring**
   ```bash
   python train.py --device auto --name strike_a      # 32 envs x 256 steps = 8192 samples per update
   tensorboard --logdir runs
   python evaluate.py --ckpt runs/strike_a/ckpt_best.pt --episodes 100   # deterministic policy
   ```
   Training success is measured with exploration noise on, so the deterministic evaluation is the real number.

8. **Behavior cloning from the oracle, then PPO fine-tuning.** PPO from scratch tends to find a reward hack: it swoops in from a steep angle and turns sharply at the goal instead of lining up behind it. Starting PPO from a clone of the scripted oracle avoids that.
   ```bash
   python bc.py --name bc                                         # DAgger, 120 rounds
   python evaluate.py --ckpt runs/bc/bc_latest.pt --episodes 100   # should approach the oracle's ~90%
   python train.py --init-from runs/bc/bc_latest.pt --name strike_bc --lr 1e-4
   ```
   - `bc.py` uses DAgger, not plain cloning. Each round, the training envs execute the oracle's action with probability `beta_decay^round` and the learner's otherwise, plus PPO's exploration noise. Every visited state is labelled with the oracle's action, and the actor is regressed on the aggregated dataset. Plain cloning only sees the oracle's states, so the learner's small errors lead it to states it never got labels for.
   - The envs are spread over curriculum levels 0 to 1 (`--levels`), because the success tolerances are part of the observation.
   - The oracle's phase (approach, then strike) is a function of the observation, not a hidden flag. With a latched flag, the same state at the lead-in point was labelled "hold" before the flag flipped and "go" after it. The clone averaged the two and stalled there (about 5% success). Instead, the strike continues while the head is in a cone around the strike line and its smoothed commanded velocity (`prev_action`) is committed forward. `Oracle(markov=False)` restores the latched version.
   - `--init-from` loads the weights and observation normalizers. The optimizer, curriculum and step count start fresh. For the first `--critic-warmup` iterations (20 by default) only the critic trains: advantages from an untrained critic would otherwise push the actor off the cloned behavior before PPO has a useful signal. A lower `--lr` than from scratch also helps keep it close.

## Command-line flags

Defaults marked `PPOCfg` come from `config.py`.

### `train.py`

| flag | default | description |
|---|---|---|
| `--device {auto,cpu,cuda}` | `auto` | Torch device; `auto` uses CUDA when available |
| `--num-envs N` | `32` (`PPOCfg.num_envs`) | Number of parallel envs |
| `--rollout-steps N` | `256` (`PPOCfg.rollout_steps`) | Steps per env per PPO iteration |
| `--total-steps N` | `200000000` (`PPOCfg.total_steps`) | Total env steps to train for |
| `--lr LR` | `3e-4` (`PPOCfg.lr`) | Learning rate at the start of the run |
| `--lr-final LR` | `3e-5` (`PPOCfg.lr_final`) | Learning rate at the end; it anneals linearly from `--lr` over this invocation (unless `PPOCfg.adaptive_lr`) |
| `--epochs N` | `5` (`PPOCfg.epochs`) | PPO epochs per update |
| `--eval-envs K` | `2` | Envs that act deterministically and stay out of the PPO batch. Their success drives the curriculum and `ckpt_best`; 0 uses the stochastic training episodes instead |
| `--grip-mode {weld,physical}` | `weld` (`EnvCfg.grip_mode`) | Weld the tool to the palm (Stage A) or grasp it physically (Stage B). Also overrides a resumed checkpoint's config |
| `--action-mode {head_twist,joint}` | `head_twist` (`EnvCfg.action_mode`) | Arm actions as a head velocity command, or as joint-target changes. Also overrides a resumed checkpoint's config |
| `--seed N` | `0` | Seed for torch, numpy and the envs (env `i` gets `seed*1000 + i`) |
| `--level L` | `0.0` | Initial curriculum level, from 0 to 1 |
| `--fixed-level` | off | Disable the automatic curriculum and stay at `--level` |
| `--serial` | off | Run envs in the main process instead of subprocesses, for debugging (always on when `--num-envs 1`) |
| `--logdir DIR` | `runs` | Parent directory for runs |
| `--name NAME` | `posevel_<timestamp>` | Run name; output goes to `<logdir>/<name>` |
| `--resume CKPT` | none | Checkpoint to resume from; its env config is used instead of `config.py`. Checkpoints from before the weld / head-twist change can't be resumed |
| `--init-from CKPT` | none | Start a new run from this checkpoint's weights and env config (for example a `bc.py` checkpoint). The optimizer, curriculum and step count start fresh. Can't be combined with `--resume` |
| `--critic-warmup N` | `20` with `--init-from`, else `0` | Iterations at the start of this invocation that train only the critic, with the actor frozen |
| `--reward-from-config` | off | On `--resume` or `--init-from`, use `RewardCfg` from `config.py` instead of the checkpoint's |
| `--save-every N` | `50` (`PPOCfg.save_every`) | Checkpoint interval in iterations |
| `--eps-p LEVEL0 LEVEL1` | `0.08 0.03` | Success position tolerance in meters at curriculum level 0 and 1. Also overrides a resumed checkpoint's config |
| `--sigma-p S` | `0.05` | Position scale in meters of the pass reward. Also overrides a resumed checkpoint's config |

### `evaluate.py`

| flag | default | description |
|---|---|---|
| `--ckpt CKPT` | none | Checkpoint to load; the env config and curriculum level are taken from it |
| `--policy {ckpt,random,zero,oracle}` | `ckpt` if `--ckpt` is given, else `random` | Which policy drives the robot. `oracle` is the scripted strike (needs `head_twist` actions) |
| `--grip-mode {weld,physical}` | from checkpoint, else `config.py` | Grip mode override |
| `--action-mode {head_twist,joint}` | from checkpoint, else `config.py` | Action mode override |
| `--episodes N` | `5` | Number of episodes |
| `--render {none,human,rgb_array}` | `none` | Render mode; `human` opens the viewer |
| `--video PATH` | none | Write an mp4 (implies `--render rgb_array`) |
| `--level L` | from checkpoint, else `0` | Curriculum level |
| `--stochastic` | off | Sample actions instead of using the mean |
| `--realtime` | off | Slow down to real time in the viewer |
| `--seed N` | `0` | Seed for the env and the random policy |
| `--device DEVICE` | `cpu` | Torch device for the policy |
| `--table-start` | off | Physical grip only: every episode starts with a new tool on the table (no in-hand starts, no soft resets); the arm still stays where the previous episode left it |

### `bc.py`

Weld grip and `head_twist` actions only (the oracle needs both). It writes `runs/<name>/bc_latest.pt` after every round.

| flag | default | description |
|---|---|---|
| `--device {auto,cpu,cuda}` | `auto` | Torch device |
| `--num-envs N` | `32` (`PPOCfg.num_envs`) | Number of parallel envs |
| `--eval-envs K` | `4` | Envs driven by the deterministic learner alone (no oracle, no noise); their success is reported as `eval succ`. Their states are still labelled and trained on |
| `--rollout-steps N` | `256` | Steps per env per DAgger round |
| `--rounds N` | `120` | DAgger rounds (about 6 minutes with 32 envs; eval success plateaus around here) |
| `--beta-decay B` | `0.8` | The training envs execute the oracle's action with probability `B^round` (round 0 is pure oracle) |
| `--noise-std S` | `exp(PPOCfg.init_log_std)` | Gaussian noise added to executed actions in the training envs |
| `--grad-steps N` | `500` | Actor gradient steps per round |
| `--batch-size N` | `4096` | Batch size for those steps |
| `--lr LR` | `1e-3` | Adam learning rate |
| `--buffer-size N` | `2000000` | Size of the aggregated dataset; the oldest samples are dropped beyond it |
| `--levels LO HI` | `0 1` | Curriculum levels the envs are spread over |
| `--seed N` | `0` | Seed |
| `--serial` | off | Run envs in the main process |
| `--logdir DIR`, `--name NAME` | `runs`, `bc_<timestamp>` | Output directory |

### `scene.py`

| flag | default | description |
|---|---|---|
| `--view` | off | Open the interactive MuJoCo viewer |
| `--seed N` | `0` | Seed for the tool and goal |
| `--level L` | `0.0` | Curriculum level (affects in-hand starts) |

## Files

| file | contents |
|---|---|
| `config.py` | Every tunable setting: scene, tool ranges, goal region, reward weights, curriculum, PPO |
| `scene.py` | Attaches the LEAP hand to the UR5e flange (`MjSpec.attach`) and adds the table, floor and goal markers. Recompiles the model per tool, with an optional palm-tool weld. |
| `tool_gen.py` | Random tool: handle (cylinder/capsule/box) + head (box/cylinder/sphere/ellipsoid/capsule) in a T, L or inline layout |
| `goals.py` | Goal sampling, either from arm configurations with the current grasp (weld) or from a precomputed reach map (physical), plus the goal-plane crossing test |
| `rewards.py` | Pass reward, approach potential, and the grasp potential for physical grip |
| `env.py` | `PoseVelocityEnv` (Gymnasium) |
| `vec_env.py` | Subprocess vector env with same-step autoreset that also returns privileged critic features |
| `ppo.py` | Asymmetric actor-critic, observation normalizers, PPO update, GAE |
| `train.py` | Training: rollouts, curriculum, TensorBoard, checkpoints |
| `evaluate.py` | Evaluation: viewer, video, success stats |
| `oracle.py` | The scripted `Oracle` strike (privileged), used by `evaluate.py --policy oracle` and as the `bc.py` expert |
| `bc.py` | DAgger behavior cloning of the oracle into the actor, as an initialization for `train.py --init-from` |

## How an episode works

An episode is one goal, which gets one attempt.

- **Grip (weld, the default):** the fingers are frozen closed. The tool is welded to the palm with its handle across the fingers, slid up to ±30% along the handle, at a random roll, and with the head on a random side. Collisions between the hand and the tool are turned off; the arm, table and floor still collide with the tool.
- **Goal:** a point `p_g`, a velocity `v_g` and a head axis `a_g`.
  - **Weld:** goals are sampled from arm configurations near the current pose. The goal is the head pose at that configuration, so it is reachable with this grasp by construction. `v_g` runs along the head axis, or within a cone of it with `GoalCfg.axis_mode = "cone"`.
  - The lead-in point (12 cm before `p_g` along the motion) and the follow-through point (8 cm after it) must each be reachable by IK with the same axis, without collisions.
  - If no goal can be found near the current pose (an awkward posture), the arm is re-homed and sampling is retried, then a new grasp is tried too. `ep/goal_recovery` logs how often that happens. The grasp-agnostic region sampler is only a last resort, logged as `ep/goal_fallback`; it has never triggered in testing. So weld goals are always reachable, and the policy can't dodge goals by parking the arm where none can be drawn.
  - **Physical:** goals come from the arm's palm reach map, as before.
- **Pass:** the attempt is the moment the head crosses the plane through `p_g` with normal `v̂`, going forward, within `pass_radius` (15 cm) of `p_g`. The pass ends the episode. Crossings farther out are ignored.
- **Success:** a pass where all three of these hold:
  - the crossing point is within `eps_p` of `p_g`;
  - `|v_head - v_g| < eps_v_frac · |v_g|`, a fraction of the goal speed, so a stationary head never qualifies;
  - the angle between the head axis and `a_g` is below `eps_theta`.

  The curriculum tightens these from 8 cm / 60% of goal speed / 60° to 3 cm / 25% of goal speed / 20°. It also raises the maximum goal speed from 0.3 to 0.6 m/s (goals are at least 0.3 m/s). Because the velocity tolerance is relative to goal speed, slow goals are the hardest to hit under exploration noise, not the easiest. The curriculum moves up at 50% success.
- **Drop (physical only):** the tool had been grasped and then either lost all hand contact for 0.2 s or touched the floor. It also counts as a drop if the tool is knocked off the table. A drop ends the episode early.
- **Time limit:** 5 s (250 control steps at 50 Hz) without a pass.
- **After an episode:**
  - **Soft reset:** the whole physics state is kept and only the goal changes. With physical grip this happens only if the tool is still held.
  - **Regrip (weld):** every `regrip_every` (8) goals, a new tool and grasp. With `regrip_rehome` (the default), the arm also returns to home, which stops it drifting into awkward postures.
  - **Respawn (physical):** if the tool isn't held, a new tool is placed on the table (or in the hand, early in the curriculum), and the arm stays where it was.
  - The arm also returns home on the first reset, on `reset(seed=...)`, and after the physics diverges.

## Model inputs and outputs

- **Action (6 values in [-1, 1] with weld; 22 with physical grip):**
  - Arm, `head_twist` (default): the head's linear velocity (up to 0.8 m/s) and angular velocity (up to 2.5 rad/s). Damped least squares maps this to joint velocities `dq`. The servo target is set to `q + (kv/kp)·dq`, which turns the UR5e's PD position servos into velocity servos.
  - Arm, `joint`: changes to the joint-position targets, up to 0.03 rad per step.
  - Hand (physical grip only): changes to 16 joint targets, up to 0.08 rad per step.
  - Actions are smoothed with an EMA (`action_ema = 0.8`). The smoothed action is what goes into the observation.
  - Exploration noise starts at std 0.135 (`init_log_std = −2`). Together with the EMA, that's about 0.05 m/s of head-velocity noise per axis. At the earlier settings (std 0.37, EMA 0.5) the noise was about 0.15 m/s per axis, and even the scripted oracle succeeded only 5% of the time with it added.
- **Actor observation (about 100 values with weld):**
  - arm joint positions and velocities, and the previous action;
  - palm pose, and the tool pose in the palm frame (the grasp);
  - tool head pose and velocity, and shape features of the tool;
  - the goal, the position, velocity and axis errors to it, and the offset to the lead-in point;
  - the current success tolerances, and time remaining.
  - Physical grip adds hand joints, fingertip positions, contact flags and grasp flags.
- **Critic-only inputs** (`info["priv"]`): tool mass, center of mass and friction; contact forces; closest distance so far; curriculum level.
- **Networks:**
  - Actor: MLP 512-256-128 (ELU) with a Gaussian output.
  - Critic: MLP 512-512-256.
  - Both use running observation normalization. They have separate Adam optimizers and separate gradient clipping.
- **Reward:** five terms, with weights in `RewardCfg`.
  - **Pass**, paid once at the crossing: `20 · q`, plus 10 if it's a success, where
    `q = exp(−d/σp) · exp(−|v_along − |v_g|| / |v_g|) · exp(−|v_side| / |v_g|) · max(0, cos θ)`.
    `v_along` and `v_side` are the head velocity's components along and across the goal direction.
    - It's a product, so a head parked on the goal earns nothing.
    - Speed error along `v̂` is scored leniently: a pass at 3× speed still keeps about 10% of that factor, so slowing down always pays.
    - **Waypoint gate** (`pass_requires_waypoint`): the pass reward and success bonus are paid only if the attempt hit the waypoint first. A pass without it counts as quality 0, so all of its approach reward is clawed back.
      - Earlier runs learned to swoop in from the side and cut across the plane at an angle, which capped success at about 25%. The gate means that can't pay, however close it gets.
      - This applies to the training reward only. `succ`, `eval succ` and the curriculum still count every pass that meets the tolerances.
      - A steep sideways penalty (`sigma_v_perp_frac = 0.25`) was tried first. It didn't stop the swoop, and it shrank the reward for good passes under exploration noise.
    - The log shows both components at passes as `(along … side …)`, as fractions of goal speed. They're `pass/v_along_frac` and `pass/v_side_frac` in TensorBoard.
  - **Approach**, paid whenever the attempt reaches a new best `D = d_lead/0.1 + θ/0.5`. Here `d_lead` is the head's distance to the lead-in point and `θ` the axis error.
    - It pays `1 · (previous best D − new D)`, so at most `D(start)` per attempt, and it's paid immediately. That makes it a real learning signal, unlike potential-based shaping.
    - **Clawback:** at a pass, `(1 − pass quality)` of what the attempt earned from approach is taken back. A drop takes all of it back. A time-out keeps it.
      - So rushing through the goal to get a fresh goal (and fresh approach reward) hands the reward back.
      - Getting to the lead-in point and waiting is a safe stepping stone, never worse than a bad pass.
  - **Waypoint:** a one-off bonus of 3 the first time in an attempt that the head is within 3 cm of the lead-in point with its axis within 15°. It counts as approach reward, so the same clawback applies. It gives a policy that swoops in from the side a concrete target for lining up behind the goal.
  - **Track**, the dense signal for the strike itself. Each step pays `10 · q · Δs / lead_in` for forward motion `Δs` along the lead-in segment.
    - Quality: `q = exp(−d_line/0.03) · exp(−|v−v_g| / (0.5·|v_g|)) · max(0, cos θ)`.
    - Only motion within 6 cm of the line, coming from behind the goal plane, counts.
    - Each attempt has a budget of `lead_in` metres of motion, so it earns at most 10. Backtracking and re-advancing just spends the budget.
    - It's paid per distance, not per step, so crawling earns nothing extra; rushing lowers `q`.
  - **Incentive check** (discounted return from attempt starts, the quantity PPO maximises), each step better than the last:

    | policy | return |
    |---|---|
    | doing nothing | 0 |
    | swooping in from the side (strike_f's / strike_g's learned policies) | +2.1 / +1.6 |
    | going to the lead-in point and waiting | +5.5 |
    | running back and forth along the line without crossing | +9.4 |
    | crawling (0.3×) or rushing (3×) through the goal | +9.2 / +9.2 |
    | the scripted oracle | +41.2 (+31.7 with training noise) |


  - **Earlier reward designs that failed:**
    - Re-paying the approach reward on every goal taught the policy to rush through goals.
    - Charging `−Φ(start)` per attempt taught it to park the arm where only unreachable goals could be drawn.
    - Without the waypoint gate, every run settled into swooping across the plane from the side.
    - Exact potential-based shaping can't be exploited, but it gives no advantage signal until real reward exists, so the approach was never learned.
  - **Contact:** −0.1 per step when the robot or tool touches the table, floor or pedestal. With physical grip, only arm links count.
  - **Physical grip adds** a grasp potential (reach, touch, lift, as `Φ' − Φ`) and a −10 drop penalty.
  - Only drops and physics divergence cut the PPO return. A pass, a time-out, a new goal or a regrip doesn't, because the world carries on.
- **PPO:** 32 envs × 256 steps per update, 5 epochs (`--epochs`) of 4 minibatches.
  - The lr anneals linearly from `--lr` to `--lr-final` over each invocation of `train.py`, so a resumed run starts at the `--lr` it's given.
  - An update stops early if KL to the rollout policy exceeds 0.05.
  - The last `--eval-envs` (default 2) envs act deterministically, using the policy mean, and their transitions stay out of the PPO batch. Their success rate (`eval succ` in the log, `eval/*` in TensorBoard) drives the curriculum and `ckpt_best`, so exploration noise doesn't hold the curriculum back.

`PoseVelocityEnv` is a standard Gymnasium env, so SB3 also works as a quick baseline: `PPO("MlpPolicy", PoseVelocityEnv())`. The SB3 baseline uses only the actor observation.

## Troubleshooting

- **`AttributeError` mentioning `MjSpec` / `attach` / `copy`:** your MuJoCo is too old. Run `pip install -U "mujoco>=3.3"`.
- **`ImportError: robot_descriptions`:** install it, or set the two `POSEVEL_*_XML` variables shown above.
- **Hand mount:** by default (`auto_align_hand=True`) the fingers are aligned with the flange axis and the palm sits flush on the flange. To change which way the palm faces, set `hand_mount_euler = (0, 0, angle)`, which spins the hand about the wrist axis. Use `hand_mount_pos` for small offsets. Re-run `python scene.py --view` to check.
- **Viewer does not open under WSL:** WSLg is needed (`echo $DISPLAY` should print something). Use `--video` as a headless alternative.
- **Headless video rendering fails:** run `export MUJOCO_GL=egl` (or `osmesa`) first.
