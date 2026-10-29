# SimToolReal baseline on the simplify-weld environment

This branch adds a baseline that trains with the **SimToolReal** method (Kedia, Lum et al., arXiv 2602.16863) in the
**simplify-weld** environment: same robot (UR5e + LEAP hand), same MuJoCo scene, same procedurally generated tools
welded to the palm. Only the training method changes, so the two approaches can be compared directly.

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
| Domain randomization | tool shape / mass / friction | optional (`--dr`): delays, pose / joint-velocity noise, random wrenches |

## Install

Same environment as `simplify-weld` (see `README.md`). Nothing extra is needed. Tested on an M-series MacBook Pro, CPU
only.

## Verify, in this order

1. **Env tests** (6 tests: spaces, random rollout, goal at the current pose is reached, delta goals are reachable,
   tolerance threshold, tool reuse):
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
4. **Evaluate the included checkpoint** (see Results):
   ```bash
   python strl_evaluate.py --ckpt checkpoints/strl_seed0/ckpt_best.pt --episodes 50 --tol 0.03
   mjpython strl_evaluate.py --ckpt checkpoints/strl_seed0/ckpt_best.pt --render human --realtime --episodes 3
   ```
   On macOS the viewer (`--render human`) needs `mjpython`; everything else runs with `python`.

## Training

```bash
# full run: 12 envs, 100M steps, ~9-10 h on a MacBook Pro
caffeinate -i python strl_train.py --name strl_seed0 --seed 0
# resume after an interruption
caffeinate -i python strl_train.py --resume runs/strl_seed0/ckpt_latest.pt
tensorboard --logdir runs
```

Checkpoints go to `runs/<name>/`: `ckpt_latest.pt` every 50 iterations and `ckpt_best.pt` (tightest tolerance, then most
goals per episode, once the curriculum window is full). A checkpoint stores the env / baseline / PPO configs, so
`strl_evaluate.py` rebuilds the same environment from it.

To reproduce the included `strl_seed0` model exactly, add `--tol-window 50` (it was trained before the default window
was raised to 200):

```bash
python strl_train.py --name strl_seed0 --seed 0 --tol-window 50
```

## Results (strl_seed0, included in `checkpoints/strl_seed0/`)

100M steps, seed 0, tolerance window 50, DR off. The curriculum reached **1.13 cm** (target 1 cm). Evaluation: 50
episodes, deterministic policy, eval seed 12345. Per-goal success = goals reached / goals attempted.

| tolerance | per-goal success | goals per episode (mean) |
|---|---|---|
| 3 cm | 83 % | 4.96 |
| 2 cm | 78 % | 3.50 |
| 1.13 cm (training) | 67 % | 2.00 |
| 1 cm | 59 % | 1.46 |
| scripted reference (`qref`), 3 cm and 1 cm | 96 % | ~20 |

Most failures are timeouts (the tool gets close but not within tolerance in 10 s); table contacts end about 7 % of
attempts. Raw outputs are in `results/strl_seed0/`.

A second run with tolerance window 200 (`strl_seed0_w200`, not included) ended at 2.12 cm and was worse on the
evaluation, so with a single seed the effect of the window is inconclusive.

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
| `--dr` | off | enable domain randomization |
| `--resume` | | continue from a checkpoint |
| `--save-every` | 50 | iterations between `ckpt_latest.pt` saves |
| `--serial` | off | run envs in the main process (debugging) |
| `--device` | cpu | `auto`, `cpu`, `cuda`, `mps` |

### strl_evaluate.py

| flag | default | meaning |
|---|---|---|
| `--ckpt` | | checkpoint to evaluate |
| `--policy` | `ckpt` if `--ckpt` else `random` | `ckpt`, `random`, `zero`, `qref` (scripted reference) |
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
| `strl_config.py` | `StrlCfg` (all baseline settings, with paper / official-code defaults), `make_env_cfg`, `make_ppo_cfg` |
| `strl_env.py` | `StrlEnv`, a subclass of `PoseVelocityEnv` that replaces the goals, reward, observations and DR |
| `strl_train.py` | PPO training loop with the tolerance curriculum |
| `strl_evaluate.py` | deterministic evaluation, scripted reference, viewer, video |
| `tests/test_strl_env.py` | env tests |
| `bench_sps.py` | env throughput benchmark |
| `checkpoints/strl_seed0/ckpt_best.pt` | trained model (see Results) |
| `results/strl_seed0/` | evaluation outputs |

## How an episode works

1. **Reset**: the arm starts at home (+-0.1 rad noise). Every 4th episode a new random tool is generated; otherwise the
   same tool is re-welded at a new random grasp.
2. **First goal**: a random tool pose in the goal region, checked reachable by IK.
3. **Steps** (50 Hz): the policy outputs arm joint-target deltas (up to 0.03 rad per step). Reward = 200 x (decrease of
   the closest keypoint distance so far) - 0.03 x arm joint speed, plus a bonus near the goal.
4. **Goal reached**: keypoint distance below `tol x 1.5` for 10 steps -> +1000, and a new goal is sampled as a random
   walk from the previous one (up to 10 cm per axis, 90 deg), with a fresh 10 s clock.
5. **End**: 50 goals reached, table / floor contact force > 100 N, or 10 s without reaching the current goal.

## Model inputs and outputs

- **Actor input (68)**: arm joint positions, velocities and current targets; palm position and orientation; fingertip
  positions relative to the palm; tool orientation; tool keypoints relative to the palm; tool keypoints minus goal
  keypoints; handle dimensions. With `--dr` these are delayed and noised.
- **Critic input (actor input + 51 privileged)**: palm and tool velocities, clean keypoints and tool orientation, closest
  distance so far, near-goal / success / time counters, last reward, tool mass / center of mass / friction, table force.
- **Output (6)**: arm joint-target deltas in [-1, 1], scaled by 0.03 rad. The fingers stay fixed (the tool is welded).
- Network: MLP 1024-1024-512-512 for actor and critic, initial action std 0.1.

## Deviations from the paper

- **No grasping stage**: the tool is welded, so the approach and lift reward terms are dropped. This is the point of the
  comparison.
- **PPO instead of SAPG** (SAPG needs thousands of parallel envs) and **MLP instead of LSTM** (for now).
- **Arm speed 1.5 rad/s** instead of 0.15 rad/s, the same as simplify-weld's joint mode (`StrlCfg.arm_delta_scale`).
  The initial action std is lowered from 1 to 0.1 so the exploration noise in joint speed matches the paper.
- **Batch and curriculum cadence** scaled down for CPU training (comments in `strl_config.py`).
- **Domain randomization off by default**: it is for sim-to-real; simplify-weld has none and both are evaluated in
  simulation.

## Troubleshooting

- `mjpython` crash with `scene.py --view`: use `python scene.py --view`; `mjpython` is only for `--render human`.
- Training stops when the Mac sleeps: run under `caffeinate -i` and keep the lid open; resume with `--resume`.
- Tolerance stops tightening around 1.1-2 cm: expected for this model size and budget; see Results.
