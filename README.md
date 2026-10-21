# pose-velocity

A UR5e arm with a LEAP hand holds a randomly generated tool. It moves the tool so that the **head center passes through a goal point, with the tool in a goal orientation, while moving at a goal velocity**. The velocity need not point out of the head's face: a goal can be a straight strike, or a swipe where the tool faces one way and moves another.

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
   - Goal markers: the **green** sphere is the goal point, the **red** arrow is the goal velocity (its length scales with speed), the **blue** arrow is the goal head axis, and the **yellow** bar runs back along the goal handle direction. The blue and yellow markers together show the full goal orientation.

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
   The oracle flies the head to the lead-in point, turns the tool to the goal orientation, then moves at `v_g` through the goal while holding that orientation. It should reach about 90% success at both levels (at level 1 that holds across swipe angles from 0° to 180°). If it falls well below that after you change the env, the env is broken, not the learner.

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
   - The envs are spread over curriculum levels 0 to 1 (`--levels`), because the success tolerances are part of the observation and the velocity cone (strikes to swipes) widens with the level.
   - Checkpoints from before the full-orientation goals (checkpoint format 2) have a different observation layout and are rejected; re-run `bc.py`.
   - The oracle's action is a continuous function of the observation. Its velocity is a blend `(1 − α)·approach + α·strike`, where the strike weight `α` ramps smoothly with progress past the lead-in point, distance from the strike line (a cone that widens towards the goal), orientation error, and whether the smoothed commanded velocity (`prev_action`) along the line is backwards. Regression averages labels that disagree on nearby states, so every hard switch in the oracle became a place the clone got stuck:
     - A latched phase flag labelled the same state at the lead-in point "hold" before the flag flipped and "go" after it. The clone stalled there (about 5% success). `Oracle(markov=False)` restores this version.
     - A hard in/out switch at the edge of the strike region (`runs/bc_swipe12`) had labels about the goal velocity apart on either side of the edge. The clone hesitated at the lead-in point (about 4× the oracle's time there), struck at about 65% of the goal speed with extra sideways velocity, and plateaued at 20–40% success while the oracle got about 90%. More DAgger data didn't help, because more labels at the edge only pin the average down.
     - Requiring the commanded velocity to reach half the goal speed within 3 cm past the lead-in point (`commit_frac = 0.5`) made it worse: a clone that accelerated slightly slower than the oracle was labelled "go back" and cycled there.
   - Watch `bc/loss_new` (loss on each round's states before training on them), not `bc/loss`. The training loss is averaged over all the data, which is mostly easy early states, and was about 10× lower than the error on the states the clone visits. `--recent-frac` draws half of each batch from the newest `--recent-rounds` rounds for the same reason.
   - The oracle-mixing schedule decays per round, so with fewer envs scale `--rounds` and `--beta-decay` to keep the total data and the oracle-guided share about the same. The training envs execute about `(training envs × rollout steps) / (1 − beta_decay)` oracle actions in total. For example, `--num-envs 24 --rounds 180 --beta-decay 0.86` gives about 1.1M samples and 37k oracle actions, close to the 32-env defaults over 150 rounds.
   - `--init-from` loads the weights and observation normalizers. The optimizer, curriculum and step count start fresh, and the reward comes from `config.py` (the fine-tuning reward), not from the checkpoint. For the first `--critic-warmup` iterations (20 by default) only the critic trains: advantages from an untrained critic would otherwise push the actor off the cloned behavior before PPO has a useful signal. A lower `--lr` than from scratch also helps keep it close.

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
| `--reward-from-config` | off | On `--resume`, use `RewardCfg` from `config.py` instead of the checkpoint's. `--init-from` always does |
| `--save-every N` | `50` (`PPOCfg.save_every`) | Checkpoint interval in iterations |
| `--eps-p LEVEL0 LEVEL1` | `0.08 0.03` | Success position tolerance in meters at curriculum level 0 and 1. Also overrides a resumed checkpoint's config |
| `--sigma-p S` | `0.03` | Position scale in meters of the pass reward. Also overrides a resumed checkpoint's config |

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
| `--lr-final LR` | `1e-4` | The learning rate is annealed linearly to this over the rounds |
| `--buffer-size N` | `2000000` | Size of the aggregated dataset; the oldest samples are dropped beyond it |
| `--recent-rounds N` | `10` | The newest N rounds of data form the "recent" pool |
| `--recent-frac F` | `0.5` | Fraction of each batch drawn from the recent pool; the rest comes from the whole dataset |
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
| `rewards.py` | Pass quality, and the grasp potential for physical grip |
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
- **Goal:** a point `p_g`, a velocity `v_g` and a full tool orientation `R_g` (a quaternion, so roll about the head axis counts too).
  - **Weld:** goals are sampled from arm configurations near the current pose. The goal is the tool pose (head position and orientation) at that configuration, so it is reachable with this grasp by construction.
  - The direction of `v_g` is drawn uniformly within a cone around the head axis at the goal pose. The cone widens with the curriculum (`vel_cone_deg`), from 30° at level 0 to 180° at level 1, where any direction is possible: sideways or even backwards swipes. Velocities pointing steeply into the table are still rejected.
  - The lead-in point (12 cm before `p_g` along the motion) and the follow-through point (8 cm after it) must each be reachable by IK in the goal orientation, without collisions.
  - If no goal can be found near the current pose (an awkward posture), the arm is re-homed and sampling is retried, then a new grasp is tried too. `ep/goal_recovery` logs how often that happens. The grasp-agnostic region sampler is only a last resort, logged as `ep/goal_fallback`; it has never triggered in testing. So weld goals are always reachable, and the policy can't dodge goals by parking the arm where none can be drawn.
  - **Physical:** goals come from the arm's palm reach map, as before.
- **Pass:** the attempt is the moment the head crosses the plane through `p_g` with normal `v̂`, going forward, within `pass_radius` (15 cm) of `p_g`. The pass ends the episode. Crossings farther out are ignored.
- **Success:** a pass where all three of these hold:
  - the crossing point is within `eps_p` of `p_g`;
  - `|v_head - v_g| < eps_v_frac · |v_g|`, a fraction of the goal speed, so a stationary head never qualifies;
  - the full rotation angle between the tool's orientation and `R_g` is below `eps_theta`.

  The curriculum tightens these from 8 cm / 60% of goal speed / 60° to 3 cm / 25% of goal speed / 20°. It also raises the maximum goal speed from 0.3 to 0.6 m/s (goals are at least 0.3 m/s), and widens the velocity cone around the head axis from 30° to 180°. Because the velocity tolerance is relative to goal speed, slow goals are the hardest to hit under exploration noise, not the easiest. The curriculum moves up at 50% success.
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
  - the goal (`p_g`, `v_g`, and `R_g` as a 6D rotation), the position and velocity errors, the orientation error as a world-frame rotation vector, and the offset to the lead-in point;
  - the current success tolerances, and time remaining.
  - Physical grip adds hand joints, fingertip positions, contact flags and grasp flags.
- **Critic-only inputs** (`info["priv"]`): tool mass, center of mass and friction; contact forces; closest distance so far; curriculum level.
- **Networks:**
  - Actor: MLP 512-256-128 (ELU) with a Gaussian output.
  - Critic: MLP 512-512-256.
  - Both use running observation normalization. They have separate Adam optimizers and separate gradient clipping.
- **Reward:** set up for fine-tuning a policy that already strikes (`bc.py`), so it is just the task, with no shaping. Weights are in `RewardCfg`.
  - **Pass**, paid once at the crossing: `20 · q`, plus 10 if it's a success, where
    `q = exp(−(d/σp + |v_head − v_g| / (σv·|v_g|) + θ/σθ))`, with `σp = 3 cm`, `σv = 0.25` and `σθ = 20°`.
    - The scales are the tightest success tolerances (level 1), so each factor is `1/e` at the edge of success. The reward keeps sharpening the strike inside the success region instead of saturating there.
    - It uses the full velocity error, so a sideways crossing costs as much as a wrong speed. It's a product, so a head parked on the goal earns almost nothing.
    - The scripted oracle scores `q ≈ 0.5` (about 18–20 per goal). Crawling (0.3×) or rushing (3×) through the goal earns under 1.
    - The log shows the velocity at passes split into `(along … side …)`, as fractions of goal speed. They're `pass/v_along_frac` and `pass/v_side_frac` in TensorBoard. With swipe goals a large `side` is no longer by itself a sign of cutting across, because it's measured against the goal velocity, not the head axis. `pass/swipe_deg` is the mean angle between the goal velocity and the goal head axis.
  - **Contact:** −0.1 per step when the robot or tool touches the table, floor or pedestal. With physical grip, only arm links count.
  - **Physical grip adds** a grasp potential (reach, touch, lift, as `Φ' − Φ`) and a −10 drop penalty. With no approach shaping, physical grip has no dense signal toward the goal, so it isn't meant to be trained from scratch with this reward.
  - Every episode end (pass, time-out, drop, divergence) cuts the PPO return, so each goal is scored on its own pass. If the return carried over into the next goal, reaching that goal sooner would pay, which means rushing through this one. A time-out can be cut too, because the time remaining is part of the observation.
  - **Earlier, from-scratch versions** of this reward added approach progress toward the lead-in point, a waypoint bonus, a waypoint gate on the pass reward, a clawback of shaping on poor passes, and a per-distance track reward along the lead-in line. These guarded against hacks a policy finds when learning from scratch, such as swooping across the plane from the side or rushing through goals. Starting from the cloned oracle, they're unnecessary (see git history).
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
