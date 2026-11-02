# SimToolReal baseline on the simplify-weld environment

This branch adds a baseline that trains with the **SimToolReal** method (Kedia, Lum et al., arXiv 2602.16863) in the
**simplify-weld** environment: same robot (UR5e + LEAP hand), same MuJoCo scene, same procedurally generated tools
welded to the palm. Only the training method changes, so the two approaches can be compared directly. The baseline
follows the paper's method **and its evaluation protocol** (keypoint distance, closed-loop goal sequences, ε = 2 cm).

It is built on top of `simplify-weld` (f31cc3f). **No existing file is modified**; everything lives in new `strl_*.py`
files that import the existing `config.py`, `env.py`, `ppo.py`, `vec_env.py`, ... modules.

| | simplify-weld | SimToolReal baseline (`strl_*.py`) |
|---|---|---|
| Robot, scene, tools, palm weld | UR5e + LEAP, MuJoCo, random tools welded to the palm | same |
| Goal | head position + velocity + tool orientation, one attempt | tool pose only, reached one after another |
| Pose distance | position / velocity / angle errors | max distance over 4 object-frame keypoints |
| Arm action | head twist (IK) | joint-position-target deltas |
| Reward | pass quality at the goal plane | 200 x keypoint progress + 1000 success bonus - 0.03 x arm joint speed |
| Training | oracle -> DAgger -> PPO fine-tune | PPO from scratch, asymmetric critic |
| Curriculum | tolerances, speed, velocity cone | success tolerance x0.9 when >= 3 goals per episode (7.5 cm -> 1 cm) |
| Evaluation | pass within position / velocity / angle tolerances | Task Progress: share of goals reached, keypoint distance < 2 cm |

## Install

Same environment as `simplify-weld` (see `README.md`). Nothing extra is needed. Tested on an M-series MacBook Pro, CPU
only.

## Verify, in this order

1. **Env tests**:
   ```bash
   pytest -q tests/test_strl_env.py
   ```
2. **Scripted reference** – drives the joints to each goal's stored IK solution. It should reach ~96 % of goals at both
   3 cm and 1 cm, which shows the goals are reachable and the success check works:
   ```bash
   python strl_evaluate.py --policy qref --episodes 20 --tol 0.03
   python strl_evaluate.py --policy qref --episodes 20 --tol 0.01
   ```
3. **Smoke train** (~1 min):
   ```bash
   python strl_train.py --num-envs 4 --total-steps 30000 --name strl_smoke
   ```
4. **Evaluate the included checkpoint** with the paper's protocol (see Results):
   ```bash
   python strl_diagnose.py --ckpt checkpoints/strl_a01_seed0/ckpt_best.pt --tol 0.02 --episodes 100
   mjpython strl_evaluate.py --ckpt checkpoints/strl_a01_seed0/ckpt_best.pt --render human --realtime --episodes 3
   ```
   On macOS the viewer (`--render human`) needs `mjpython`; everything else runs with `python`.

## Training

```bash
# the included baseline: 12 envs, 100M steps, ~10 h on a MacBook Pro
caffeinate -i python strl_train.py --name strl_a01_seed0 --seed 0
# resume after an interruption
caffeinate -i python strl_train.py --resume runs/strl_a01_seed0/ckpt_latest.pt
tensorboard --logdir runs
```

Checkpoints go to `runs/<name>/`: `ckpt_latest.pt` every 50 iterations and `ckpt_best.pt` (tightest tolerance, then most
goals per episode, once the 200-episode curriculum window is full). A checkpoint stores the env / baseline / PPO configs,
so the evaluation scripts rebuild the same environment from it.

The earlier model `checkpoints/strl_seed0/` was trained with the old defaults; to reproduce it add
`--arm-delta-scale 0.03 --tol-window 50`.

## Evaluation protocol

As in the paper (Sec. IV and the appendix): **Task Progress** = the share of goals reached, where a goal counts as reached
when the keypoint distance d(o, g) is below **ε = 2 cm**, evaluated **closed-loop** (the next goal appears only after the
current one is reached; 10 s per goal). The paper uses goal sequences from DexToolBench demonstrations; we use the
training goal distribution (a random first goal, then random walks of up to 10 cm and 90 deg). Deterministic policy
(the mean action), no domain randomization, 100 episodes, seeds 12345 + i.

`strl_diagnose.py --tol 0.02` reports this number ("reached") and breaks the failures down by goal type, rotation,
distance, height above the table and tool.

## Results

### Baseline: `strl_a01_seed0` (included in `checkpoints/strl_a01_seed0/`)

100M steps, seed 0, arm action 0.01 rad per step, curriculum window 200, DR off. The curriculum reached **1.25 cm**.

| tolerance ε | Task Progress (per-goal success) | goals per episode | episodes ended by a table contact |
|---|---|---|---|
| 3 cm | 87 % | 6.9 | 2 % of goals |
| **2 cm (paper)** | **83 %** | **4.8** | 3 % of goals |
| 1.25 cm (training) | 71 % | – | 2 % of goals |
| 1 cm | 60 % | 1.5 | 2 % of goals |
| scripted reference (`qref`), 3 cm and 1 cm | 96 % | ~20 | |

Success falls with the rotation a goal needs (2 cm): 97 % for 0–30 deg, 89 % for 30–60, 73 % for 60–90, 41 % for
90–120. With the tool welded there is no in-hand rotation, so large rotations need large arm motions; failures there
are mostly the policy stopping just outside the tolerance, and table contacts are mostly the wrist / forearm during
those motions. The paper's Fixed Grasp baseline shows the same failure ("arm collides with table during rotation").
Raw outputs: `results/strl_a01_seed0/`.

### Earlier model: `strl_seed0` (included in `checkpoints/strl_seed0/`)

Arm action 0.03, curriculum window 50; reached 1.13 cm. Task Progress 85 % (3 cm), 78 % (2 cm), 59 % (1 cm). The window
of 50 episodes let the tolerance tighten on lucky batches; with window 200 and the same action scale the run stopped at
2.12 cm (77 % / 21 % at 3 / 1 cm). Raw outputs: `results/strl_seed0/`.

### Screening runs (how the action scale was chosen)

20M steps each, seed 0, curriculum window 200; Task Progress of `ckpt_latest.pt` at 3 cm / 5.5 cm (100 episodes).

| run | arm action (rad / step) | table penalty | 3 cm | 5.5 cm |
|---|---|---|---|---|
| `scr2_base` | 0.03 | – | 28 % | 74 % |
| `scr2_pen` | 0.03 | 3 per contact step + 200 on a table termination | 17 % | 60 % |
| `scr2_pen_a01` | 0.01 | same | 39 % | 75 % |
| **`scr2_a01`** | **0.01** | – | **48 %** | **77 %** |

The slower arm helps; the table penalty (not in the paper) made the policy avoid large rotations and is left off
(`--table-penalty` / `--contact-penalty` remain available, default 0). A run with a fixed 3 cm tolerance from the start
(no curriculum) still completed only ~0.4 goals per episode during training after 20M steps, so the curriculum
matters. Raw outputs: `results/screening/`.

### Note on the metric

The reward keypoints are a thin box (±10.6 cm along the handle, about ±2 cm across), so the keypoint distance barely
changes when the tool rolls about its handle: at ε = 3 cm a roll of up to ~95 deg still counts as reached (~28 deg at
1 cm), while tilting the handle is limited to ~24 deg (~8 deg). The policy therefore does not learn to match roll. This
is a property of the paper's metric. If a comparison needs a full-orientation criterion,
`strl_diagnose.py --metric weld --eps-p 0.03 --eps-theta 20` evaluates simplify-weld's head-position + full-angle test
(the baseline reaches 56 % there; 95 % of its angle failures are mostly roll), but it is not the baseline's own
evaluation.

## Command-line flags

### strl_train.py

| flag | default | meaning |
|---|---|---|
| `--name` | timestamp | run directory under `--logdir` |
| `--logdir` | `runs` | |
| `--seed` | 0 | |
| `--num-envs` | 12 | parallel env processes |
| `--rollout-steps` | 256 | steps per env per PPO update |
| `--total-steps` | 100M | |
| `--lr` | 1e-4 | initial learning rate (KL-adaptive) |
| `--lr-max` | 3e-4 | upper bound for the adaptive learning rate |
| `--epochs` | 2 | PPO epochs per update |
| `--torch-threads` | 6 | CPU threads for the PPO update |
| `--tol` | 0.075 | initial success tolerance (m) |
| `--fixed-tol` | off | disable the tolerance curriculum |
| `--tol-window` | 200 | episodes averaged by the curriculum |
| `--arm-delta-scale` | 0.01 | rad of joint-target change per step at action 1; the initial action std is rescaled with it |
| `--table-penalty` | 0 | reward added when a table contact ends the episode (not in the paper) |
| `--contact-penalty` | 0 | reward added on every step with a table / floor contact (not in the paper) |
| `--dr` | off | enable domain randomization |
| `--resume` | | continue from a checkpoint |
| `--save-every` | 50 | iterations between `ckpt_latest.pt` saves |
| `--serial` | off | run envs in the main process (debugging) |
| `--device` | cpu | `auto`, `cpu`, `cuda`, `mps` |

### strl_diagnose.py

| flag | default | meaning |
|---|---|---|
| `--ckpt` | | checkpoint to evaluate |
| `--policy` | `ckpt` if `--ckpt` else `qref` | `ckpt` or `qref` (scripted reference) |
| `--tol` | the checkpoint's | success tolerance ε (m); 0.02 for the paper's protocol |
| `--episodes` | 100 | |
| `--seed` | 12345 | episode `i` uses seed `seed + i` |
| `--csv` | | write one row per goal |
| `--metric` | `keypoint` | `weld`: simplify-weld's head position + full angle test instead |
| `--eps-p` / `--eps-theta` / `--hold` | 0.03 / 20 / 10 | `weld` metric tolerances (m, deg) and steps within them |

### strl_evaluate.py

| flag | default | meaning |
|---|---|---|
| `--ckpt` | | checkpoint to evaluate |
| `--policy` | `ckpt` if `--ckpt` else `random` | `ckpt`, `random`, `zero`, `qref` |
| `--episodes` | 10 | |
| `--tol` | the checkpoint's | success tolerance (m) |
| `--stochastic` | off | sample actions instead of the mean |
| `--dr` / `--no-dr` | the checkpoint's | force DR on / off |
| `--render` | none | `human` (needs `mjpython` on macOS), `rgb_array` |
| `--video` | | write an mp4 |
| `--realtime` | off | slow the viewer to real time |
| `--seed` | 12345 | episode `i` uses seed `seed + i` |

## Files

| file | contents |
|---|---|
| `strl_config.py` | `StrlCfg` (all baseline settings, with paper / official-code defaults and marked deviations), `make_env_cfg`, `make_ppo_cfg` |
| `strl_env.py` | `StrlEnv`, a subclass of `PoseVelocityEnv` that replaces the goals, reward, observations and DR |
| `strl_train.py` | PPO training loop with the tolerance curriculum |
| `strl_diagnose.py` | evaluation with the paper's protocol, per-goal failure breakdown |
| `strl_evaluate.py` | quick evaluation, viewer, video |
| `tests/test_strl_env.py` | env tests |
| `bench_sps.py` | env throughput benchmark |
| `checkpoints/strl_a01_seed0/ckpt_best.pt` | the baseline model |
| `checkpoints/strl_seed0/ckpt_best.pt` | the earlier model |
| `results/` | evaluation outputs of both models and of the screening runs |

## How an episode works

1. **Reset**: the arm starts at home (±0.1 rad noise). Every 4th episode a new random tool is generated; otherwise the
   same tool is re-welded at a new random grasp.
2. **First goal**: a random tool pose in the goal region, checked reachable by IK.
3. **Steps** (50 Hz): the policy outputs arm joint-target deltas (up to 0.01 rad per step). Reward = 200 x (decrease of
   the closest keypoint distance so far) - 0.03 x arm joint speed (L1), plus +100 on every step within tolerance.
4. **Goal reached**: keypoint distance below `tol x 1.5` for 10 steps (so +1000 in total), and a new goal is sampled as
   a random walk from the previous one (up to 10 cm per axis, 90 deg), with a fresh 10 s clock.
5. **End**: 50 goals reached, table / floor contact force > 100 N, or 10 s without reaching the current goal.

## Model inputs and outputs

- **Actor input (68)**: arm joint positions, velocities and current targets; palm position and orientation; fingertip
  positions relative to the palm; tool orientation; tool keypoints relative to the palm; tool keypoints minus goal
  keypoints; handle dimensions. With `--dr` these are delayed and noised.
- **Critic input (actor input + 51 privileged)**: palm and tool velocities, clean keypoints and tool orientation, closest
  distance so far, near-goal / success / time counters, last reward, tool mass / center of mass / friction, table force.
- **Output (6)**: arm joint-target deltas in [-1, 1], scaled by `arm_delta_scale`. The fingers stay fixed (the tool is
  welded).
- Network: MLP 1024-1024-512-512 for actor and critic; initial action std 0.3 (joint-speed noise ~0.15 rad/s).

## Deviations from the paper

- **No grasping stage**: the tool is welded, so the approach and lift reward terms are dropped. This is the point of the
  comparison.
- **PPO instead of SAPG** (SAPG needs thousands of parallel envs) and **MLP instead of LSTM**.
- **Arm speed 0.5 rad/s** instead of 0.15 rad/s: without in-hand rotation, ~20 % of goals need a joint to move ≥ 90 deg,
  which 0.15 rad/s cannot do within 10 s.
- **Batch and curriculum cadence** scaled down for CPU training (12 envs; the curriculum averages the last 200 finished
  episodes; comments in `strl_config.py`).
- **Domain randomization off by default**: it is for sim-to-real; simplify-weld has none and both are evaluated in
  simulation.
- **Evaluation goals** come from the training goal distribution, not DexToolBench demonstrations.

## Troubleshooting

- `mjpython` crash with `scene.py --view`: use `python scene.py --view`; `mjpython` is only for `--render human`.
- Training stops when the Mac sleeps: run under `caffeinate -i` and keep the lid open; resume with `--resume`.
- Do not start a run at a tight fixed tolerance (`--fixed-tol --tol 0.03`): without the curriculum the policy rarely
  earns the success bonus and learns very slowly.
