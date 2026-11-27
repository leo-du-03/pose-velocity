# pose-velocity

A UR5e arm with a LEAP hand holds a randomly generated tool and moves it so that the **head center passes through a
goal point, in a goal orientation, at a goal velocity**. The velocity need not point out of the head's face, so a goal
can be a straight strike or a swipe where the tool faces one way and moves another.

- **Stage A (default, `grip_mode = "weld"`):** the tool is welded to the palm at a random grasp and the policy controls
  only the arm.
- **Stage B (`grip_mode = "physical"`):** the hand must grasp the tool from the table and hold it. This works but is
  not tuned.

MuJoCo runs on CPU in subprocess workers; the policy is asymmetric PPO in PyTorch (CPU or CUDA). The recommended
pipeline is: a scripted oracle, then DAgger behavior cloning of it (`bc.py`), then PPO fine-tuning (`train.py
--init-from`). Trained policies are then evaluated on downstream tasks (hammering, sweeping, slicing, flicking).

## Install

```bash
conda env create -f environment.yml        # creates env "pose-velocity" (Python 3.10)
conda activate pose-velocity
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: smaller CPU-only torch
```

For a GPU under WSL, install the NVIDIA Windows driver, restart WSL (`wsl --shutdown`), check `nvidia-smi`, then
install a CUDA build of torch from pytorch.org.

Robot models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie): `robot_descriptions`
clones them into `~/.cache/robot_descriptions` on first use (needs `git` and internet). Offline, clone Menagerie
yourself and set:

```bash
export POSEVEL_UR5E_XML=/path/to/mujoco_menagerie/universal_robots_ur5e/ur5e.xml
export POSEVEL_LEAP_XML=/path/to/mujoco_menagerie/leap_hand/right_hand.xml
```

## Quick start

Run everything from the project root.

```bash
python scene.py              # 6 arm + 16 hand joints, 4 fingertips (thumb last); action dim 6, obs dim ~104 (weld)
python scene.py --view       # viewer; markers: green = goal point, red = goal velocity, blue = goal head axis,
                             #   yellow = goal handle direction
pytest -q tests
python evaluate.py --policy oracle --episodes 100            # scripted oracle, level 0
python evaluate.py --policy oracle --episodes 100 --level 1  # tightest tolerances, fastest goals, full swipe cone
```

The oracle succeeds on about 75–80% of goals at both levels (100 episodes each, default tolerances). If it falls well
below that after an env change, the env is broken, not the learner.

**Train** (BC, then PPO fine-tuning):

```bash
python train.py --device cpu --num-envs 4 --total-steps 20000   # smoke test, under a minute
python bc.py --name bc                                          # DAgger from the oracle, ~6 min on 32 envs
python evaluate.py --ckpt runs/bc/bc_latest.pt --episodes 100   # should approach the oracle
python train.py --init-from runs/bc/bc_latest.pt --name strike_bc --lr 1e-4
tensorboard --logdir runs
python evaluate.py --ckpt runs/strike_bc/ckpt_best.pt --episodes 100
```

- PPO from scratch tends to find a reward hack (swooping in steeply and turning at the goal); starting from the cloned
  oracle avoids it. [Trajectory goals](#trajectory-goals) are the from-scratch alternative.
- Training success includes exploration noise; the deterministic `evaluate.py` number is the real one. `ckpt_best.pt`
  is saved when the deterministic eval envs beat the previous best at the same or a higher level.
- In `bc.py`, watch `bc/loss_new` (loss on each round's new states), not `bc/loss`. The oracle's design and the DAgger
  schedule are explained in the `oracle.py` and `bc.py` docstrings.
- `--init-from` loads weights, normalizers and env config; the optimizer, curriculum and steps start fresh, the reward
  comes from `config.py`, and the first `--critic-warmup` (20) iterations train only the critic.

**Evaluate on tasks:**

```bash
python evaluate_tasks.py --task plate --policy trace --episodes 20     # the reference trace follower
python evaluate_tasks.py --task nail --ckpt runs/strike_bc/ckpt_best.pt --episodes 50 --video nail.mp4
```

## How it works

An episode is one goal, which gets one attempt. All values below are defaults in `config.py`.

**Grip (weld).** The fingers are frozen closed and the tool is welded to the palm with its handle across the fingers,
slid up to `grasp_slide` (±70%) of the handle's half length from its center, at a random roll, head on a random side.
Hand–tool collisions are off; the arm, table and floor still collide with the tool. Every `regrip_every` (8) goals
brings a new tool and grasp, and the arm returns home.

**Goal.** A point `p_g`, a velocity `v_g` and a full tool orientation `R_g`.
- Weld goals are the tool pose at a sampled arm configuration near the current one, so they are reachable with this
  grasp by construction. Physical-grip goals come from a precomputed palm reach map.
- `v_g` is drawn within a cone around the head axis that widens with the curriculum (`vel_cone_deg`, 30° → 180°, so
  sideways and backwards swipes appear). A quarter of goals (`down_frac`) are steep downward strikes, as for a nail.
- The lead-in point (12 cm before `p_g`) and the follow-through point (8 cm after) must be reachable and
  collision-free in the goal orientation. If no goal can be found, the arm re-homes and/or regrips
  (`ep/goal_recovery`).

**Attempt and success.** The attempt is the first step where the keypoint distance `d` (max distance between 4
keypoints on the tool and the same keypoints at the goal pose, so it covers position and orientation) is within
`eps_kp`. The episode ends there. It is a success if the head is within `eps_p` of `p_g`, the rotation to `R_g` is
under `eps_theta`, and `|v_head − v_g| < eps_v_frac · |v_g|` (so a stationary head never qualifies). The curriculum
moves up at 50% success, tightening the tolerances from 8 cm / 60° / 60% of goal speed to 3 cm / 20° / 25%,
raising the max goal speed from 0.3 to 0.6 m/s and widening the velocity cone.

**Other episode ends.** A 5 s time-out (250 steps at 50 Hz); the robot or tool pushing on the table or floor harder
than 100 N; in `head_twist`, the arm being unable to follow the commanded twist (`ik_fail_tol`); physics divergence;
for physical grip, a drop (`ep/dropped`). After an episode the physics state is kept and only the goal changes (a
soft reset), unless it is time to regrip or the tool was lost.

**Actions** (6 values in [-1, 1] with weld; 22 with physical grip), smoothed by an EMA (`action_ema = 0.8`):
- `head_twist` (default): the head's linear and angular velocity (0.8 m/s, 2.5 rad/s at 1), mapped to joint velocities
  by damped least squares; the servo target `q + (kv/kp)·dq` turns the PD position servos into velocity servos.
- `joint`: joint-target changes of up to `arm_action_scale` (0.03 rad) per step.
- `joint_vel`: joint velocities up to `joint_v_max`, through the same velocity servo, with no IK at runtime. Its
  observation adds the goal errors mapped through the head-Jacobian pseudo-inverse (`kin_obs`). For a joint-space
  policy, this is the mode that clones well.
- Physical grip adds 16 hand joint-target changes.

**Observation.** Actor: arm joints and velocities, the smoothed previous action, palm pose, the grasp (tool pose in the
palm frame), head pose and velocity, tool shape features, the goal (`p_g`, `v_g`, `R_g` as a 6D rotation) and its
errors, the offset to the lead-in point, the current tolerances and the time remaining. Physical grip adds hand joints,
fingertips and contact flags. The critic also gets privileged state (`info["priv"]`): tool mass, center of mass and
friction, contact forces, the closest distance so far and the curriculum level.

**Reward** (`RewardCfg`, `rewards.py`): SimToolReal's reward extended with the goal velocity.
- Keypoint progress per step: `200 · max(d* − d, 0)`, where `d*` is the closest so far, so a windup costs nothing.
- Smoothness per step: `−0.03 · Σ|q̇_arm|` (`--smooth-norm` chooses L1, L2 or L2²).
- Attempt bonus, once: `1000 · exp(−|v_head − v_g| / (0.25·|v_g|))`, so arriving slowly or sideways earns little.
- Physical grip adds a grasp potential (reach, touch, lift) and a drop penalty.
- Every episode end cuts the PPO return, so each goal is scored on its own attempt and rushing through one to reach
  the next doesn't pay.

**Networks and PPO.** Actor MLP 512-256-128 (ELU, Gaussian), critic MLP 512-512-256, running observation
normalization. 32 envs × 256 steps per update, 5 epochs of 4 minibatches, KL early stop at 0.05, lr annealed linearly
from `--lr` to `--lr-final` over each invocation. The last `--eval-envs` (2) envs act deterministically, stay out of
the PPO batch, and drive the curriculum and `ckpt_best`.

## Variants

### Trajectory goals

`--goal-mode trajectory` (`EnvCfg.goal_mode`, settings in `TrajCfg`) lets PPO learn from scratch, without behavior
cloning, by giving it a path into the strike. The goal and success rule are unchanged; each episode also gets a smooth
random reference, one sample per control step.
- It is built in joint space, so every sample is reachable: a quintic from the current joints to the start of the
  lead-in plus a random detour (`bump_rad`, growing with the level), then the lead-in through `p_g` at `v_g` and the
  follow-through. References are checked against joint limits, `joint_v_max`, collisions and table clearance, and
  their approach stays 1.5 × `eps_kp` from the goal pose. If none passes, the last resort is an unchecked reference
  (`ep/ref_fallback`).
- The actor also observes the reference now and `preview` steps ahead, and the joint reference (`joint_obs`).
- The keypoint progress term becomes a tracking reward, `w_pos·exp(−d_ref/σ_pos) + w_vel·exp(−|v − v_ref| /
  (σ_v·|v_g|))`. An episode also ends as `lost` when `d_ref > lost_dist` (15 cm).
- Weld grip, pose + velocity goals only, with `joint_vel` or `head_twist` actions. `--policy oracle` here is
  `traces.TraceFollower` on the reference (about 96% success in `joint_vel` at level 0).

```bash
python evaluate.py --policy oracle --goal-mode trajectory --action-mode joint_vel --episodes 100
python train.py --device auto --name traj_jv --goal-mode trajectory --action-mode joint_vel
```

### Pose-only baselines and goal-velocity ablations

These take the pose + velocity goal apart. All settings are saved in the checkpoint. Flags that change the observation
(`--pose-only`, `--no-vel-obs`, `--pose-chain`, `--goal-mode`) must be set in `bc.py` (or a from-scratch `train.py`);
`train.py --init-from` inherits them.

| flag | what changes |
|---|---|
| `--pose-only hold` | No goal velocity in the observation or reward. Success after 10 (not necessarily consecutive) steps within tolerance, as in SimToolReal. Expert: `oracle.PoseOracle` |
| `--pose-only touch` | As above, but the first step within tolerance succeeds and ends the episode |
| `--no-vel-obs` | Velocity goal and reward kept, but the actor gets the pose-only observation (`v_g` goes to the critic) |
| `--no-vel-reward` | `v_g` observed, but the attempt bonus is flat; success still needs the velocity |
| `--pose-chain N` | With `touch`: the velocity spelled out as N timed pose sub-goals along the strike line after the lead-in point. Expert: `oracle.ChainPoseOracle` |

`evaluate.py` scores every policy by the same rule at the attempt (pose, velocity, and both as `joint_ok`), so a
pose-only policy's velocity at the attempt shows what a pose goal gives for free. `--dump` writes per-episode results
for `analyze_ablation.py`. `eval_attempt_reward.py` compares checkpoints on the reward earned up to the attempt.

```bash
python bc.py --name bc_pose_hold --pose-only hold
python train.py --init-from runs/bc_pose_hold/bc_latest.pt --name strike_bc_pose_hold --lr 1e-4
python evaluate.py --ckpt runs/strike_bc_pose_hold/ckpt_best.pt --level 1 --episodes 200 --dump results/hold_s0.json
python analyze_ablation.py results/ --bins
```

### SimToolReal baseline

`strl_*.py` reproduce SimToolReal (pose goals reached one after another, joint-delta actions, PPO from scratch) in
the same scene. See [STRL_BASELINE.md](STRL_BASELINE.md); a trained checkpoint is in `checkpoints/strl_seed0/`.

## Task evaluation

`tasks.py` builds task envs on top of `PoseVelocityEnv`, each with a fixed tool and fixtures (settings in `TaskCfg`):

| task | tool | goal | success |
|---|---|---|---|
| `plate` | hammer | hit a plate with a target peak impact force | peak force within 15% of the target |
| `nail` | hammer | hit a vertical nail straight down with a target force | the face hits the nail head first, within 1.2 cm of its axis, force within 15% |
| `sweep` | brush with a spring bristle pad | scrub a dirty patch; stronger dirt needs more force | at least 95% of the dirt removed |
| `slice` | kitchen knife | saw a loaf in half along a marked line | cut through every column, on the line, without crushing the loaf or jamming the knife |
| `slice_tomato` | kitchen knife | slice a cherry tomato in half | cut through with the pressing force between the cut and crush forces, near the middle |
| `flick` | spatula | flick an egg off the spatula into a bowl | the egg comes to rest in the bowl |

- Each episode samples an instance and a **trace** (head pose, velocity and angular velocity per control step,
  `traces.Trace`) that achieves the task when followed exactly. Instances are calibrated and checked by running the
  reference follower (`--policy trace`, `traces.TraceFollower`), and resampled if it fails.
- Policies only know single goals, so the env feeds them the trace's **keypoints** one at a time as ordinary goals
  (the impact for hammering, plus two windup goals; waypoints along strokes for sweeping and slicing). It advances
  when the head crosses the current keypoint's goal plane.
- Tools are held as a person would hold them (handle across the fingers, the face or edge pointing the way the
  knuckles do), not at the random training grasps.
- `--layout in_dist` (default) places fixtures within the training goal region; `realistic` puts everything on the
  tabletop, which is out of distribution for current policies.
- The failure rules and physics of each task are documented in the task class docstrings in `tasks.py`.

## Common flags

Run `python <script> --help` for the full list; defaults live in `config.py`.

| flag | scripts | |
|---|---|---|
| `--device {auto,cpu,cuda}` | train, bc, evaluate* | `auto` uses CUDA when available |
| `--num-envs N`, `--name`, `--logdir` | train, bc | parallel envs; output goes to `<logdir>/<name>` |
| `--init-from CKPT` / `--resume CKPT` | train | start a new run from a checkpoint (e.g. BC) / continue a run |
| `--action-mode {head_twist,joint,joint_vel}` | train, bc, evaluate | arm action space |
| `--goal-mode {points,trajectory}` | train, bc, evaluate | see [Trajectory goals](#trajectory-goals) |
| `--pose-only`, `--no-vel-obs`, `--no-vel-reward`, `--pose-chain`, `--vel-cone` | train, bc (evaluate: all but the `--no-vel-*` pair) | see [ablations](#pose-only-baselines-and-goal-velocity-ablations) |
| `--level L` | train, evaluate* | curriculum level (0–1); train's `--fixed-level` disables the curriculum |
| `--eps-p`, `--eps-kp`, `--eps-v-frac`, `--sigma-v-frac` | train (bc: last two) | tolerances at levels 0 and 1, and the bonus's velocity scale |
| `--ckpt CKPT`, `--policy` | evaluate* | the env config and level come from the checkpoint |
| `--episodes`, `--render human`, `--realtime`, `--video PATH` | evaluate* | `--video` renders headless to mp4 |
| `--dump PATH` | evaluate | per-episode JSON for `analyze_ablation.py` |
| `--task`, `--layout`, `--save-traces DIR` | evaluate_tasks | plus per-task overrides of `TaskCfg` fields |

## Files

| file | contents |
|---|---|
| `config.py` | Every tunable setting (dataclasses), plus the shared ablation flags |
| `scene.py` | Builds the scene: LEAP hand on the UR5e flange, table, goal markers, the tool, optional palm weld |
| `tool_gen.py` | Random tools (handle + head in a T, L or inline layout) and fixed task tools |
| `goals.py` | Goal sampling (from arm configurations, or the reach map) and the goal-plane crossing test |
| `rewards.py` | Keypoint distance, velocity match, smoothness, trajectory tracking reward, grasp potential |
| `env.py` | `PoseVelocityEnv` (Gymnasium) |
| `mathutils.py` | Quaternion and rotation helpers |
| `vec_env.py` | Subprocess vector env with same-step autoreset and privileged critic features |
| `ppo.py` | Asymmetric actor-critic, observation normalizers, PPO update, GAE |
| `oracle.py` | Scripted experts: `Oracle` (strike), `PoseOracle`, `ChainPoseOracle` |
| `bc.py` | DAgger behavior cloning of an oracle into the actor |
| `train.py` | PPO training: rollouts, curriculum, TensorBoard, checkpoints |
| `evaluate.py` | Evaluation: success stats, viewer, video, `--dump` |
| `traces.py` | Reference traces, their builders, and `TraceFollower` |
| `tasks.py`, `evaluate_tasks.py` | Task envs and their evaluation script |
| `analyze_ablation.py` | Mean ± std tables over `evaluate.py --dump` files, optionally binned |
| `eval_attempt_reward.py` | Reward up to the attempt, compared across checkpoints |
| `bench_sps.py` | Simulation throughput benchmark |
| `strl_*.py`, `STRL_BASELINE.md` | The SimToolReal baseline |
| `checkpoints/`, `results/` | The SimToolReal checkpoint, its evaluation outputs, and ablation dumps |
| `tests/` | pytest suite |

## Troubleshooting

- **`AttributeError` mentioning `MjSpec` / `attach` / `copy`:** MuJoCo is too old: `pip install -U "mujoco>=3.3"`.
- **`ImportError: robot_descriptions`:** install it, or set the two `POSEVEL_*_XML` variables above.
- **Hand mount:** by default (`auto_align_hand`) the fingers continue along the flange axis, palm flush on it.
  `SceneCfg.hand_mount_euler = (0, 0, angle)` spins the hand about the wrist; `hand_mount_pos` adds small offsets.
  Check with `python scene.py --view`.
- **Viewer doesn't open under WSL:** WSLg is needed (`echo $DISPLAY` should print something); use `--video` instead.
- **Headless video rendering fails:** `export MUJOCO_GL=egl` (or `osmesa`).
