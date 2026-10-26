# SimToolReal baseline on the simplify-weld environment

This branch adds a baseline that trains with the **SimToolReal** method (Kedia, Lum et al., arXiv 2602.16863) in the
**simplify-weld** environment, so the two training approaches can be compared on the same robot, tools and scene.

| | simplify-weld (this repo) | SimToolReal baseline (`strl_*.py`) |
|---|---|---|
| Robot, scene, tools, palm weld | UR5e + LEAP, MuJoCo, random tools welded to the palm | same |
| Goal | head position + velocity + tool orientation, one attempt | tool pose only, reached one after another |
| Pose distance | position / velocity / angle errors | max distance over 4 object-frame keypoints |
| Arm action | head twist (IK) | joint-position-target deltas |
| Reward | pass quality at the goal plane | 200 x keypoint progress + 1000 success bonus - 0.03 x arm joint speed |
| Training | oracle -> DAgger -> PPO fine-tune | PPO from scratch, asymmetric critic |
| Curriculum | tolerances, speed, velocity cone | success tolerance x0.9 when >= 3 goals per episode (7.5 cm -> 1 cm) |
| Domain randomization | tool shape / mass / friction | + obs / action / object-pose delays, pose and joint-velocity noise, random wrenches |

## Files

| file | contents |
|---|---|
| `strl_config.py` | `StrlCfg` (all baseline settings, with paper / official-code defaults), `make_env_cfg`, `make_ppo_cfg` |
| `strl_env.py` | `StrlEnv`, a subclass of `PoseVelocityEnv` that replaces the task, reward, observations and DR |
| `strl_train.py` | PPO training loop with the tolerance curriculum |
| `strl_evaluate.py` | deterministic evaluation, viewer, video |
| `tests/test_strl_env.py` | env tests |

No existing file is modified.

## Usage

```bash
pytest -q tests/test_strl_env.py
python strl_train.py --num-envs 4 --total-steps 30000 --name strl_smoke    # smoke test
python strl_train.py --name strl_seed0 --seed 0                           # real run (12 envs, 100M steps)
tensorboard --logdir runs
python strl_evaluate.py --ckpt runs/strl_seed0/ckpt_best.pt --episodes 50
python strl_evaluate.py --policy qref --episodes 20      # scripted check: drives the joints to each goal's IK solution
```

## Deviations from the paper

- **No grasping stage**: the tool is welded, so the approach and lift reward terms are dropped (they only apply before
  the object is grasped). This is the point of the comparison.
- **PPO instead of SAPG**: SAPG splits thousands of parallel envs into blocks of 4096; with ~12 CPU envs it doesn't apply.
- **MLP instead of LSTM** (for now).
- **Arm speed**: the paper's arm pipeline caps joint speed at 0.15 rad/s; we use 1.5 rad/s (0.03 rad per 50 Hz step),
  the same as simplify-weld's joint mode (`StrlCfg.arm_delta_scale`). The initial action std is lowered from 1 to 0.1
  so the joint-speed exploration noise stays at the paper's ~0.15 rad/s (at std 1, PPO only learned to lift the arm
  away from the table).
- **Batch / curriculum cadence** scaled down for CPU training (see comments in `strl_config.py`).
- **Domain randomization off by default** (`--dr` enables it): it is for sim-to-real transfer, simplify-weld has none,
  and both are evaluated in simulation. Table-height randomization and robot-model calibration are not ported.
