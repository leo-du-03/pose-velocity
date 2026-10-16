# pose-velocity

A UR5e arm with a LEAP hand has to grasp a randomly generated tool. It then moves the tool so that the **head center passes through a goal point, with the head axis aligned to a goal direction, while moving at a goal velocity**.

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
   - `action dim: 22`
   - an obs dim of about 160

   The first run downloads Menagerie and builds the reach map, which takes a few seconds.

2. **Look at the scene**
   ```bash
   python scene.py --view
   ```
   - Check that the hand is mounted sensibly on the flange.
   - The tool should be lying on the table in front of the arm.
   - Goal markers: the **green** sphere is the goal point, the **red** arrow is the goal velocity (its length scales with speed), and the **blue** arrow is the goal head axis.

   If the hand is rotated or offset wrongly, tune `SceneCfg.hand_mount_euler` and `SceneCfg.hand_mount_pos` in `config.py`.

3. **Unit and integration tests**
   ```bash
   pytest -q tests
   ```
   The tests cover:
   - joint and fingertip discovery;
   - 50 random tools compile within the mass bounds and settle on the table;
   - goal constraints;
   - the Gymnasium `check_env`;
   - a 1500-step random rollout stays finite;
   - soft and respawn reset semantics (the arm stays put, the tool or goal changes);
   - a scripted pass-through of the goal triggers success.

4. **Watch the env live with random actions**
   ```bash
   python evaluate.py --policy random --render human --realtime --episodes 3
   ```
   You can also write a video instead: `python evaluate.py --policy zero --episodes 2 --video zero.mp4`

5. **Training smoke test** (runs about 1–2 minutes on CPU)
   ```bash
   python train.py --device cpu --num-envs 4 --total-steps 20000
   ```
   - The output should show iterations with `sps`, return, success and grasp rates.
   - It saves `runs/<name>/ckpt_latest.pt`.
   - `--device auto` uses CUDA when available.

6. **Real training run and monitoring**
   ```bash
   python train.py --device auto --num-envs 16
   tensorboard --logdir runs
   python evaluate.py --ckpt runs/<name>/ckpt_latest.pt --render human --realtime
   ```

## Files

| file | contents |
|---|---|
| `config.py` | Every tunable setting: scene, tool ranges, goal region, reward weights, curriculum, PPO |
| `scene.py` | Attaches the LEAP hand to the UR5e flange (`MjSpec.attach`) and adds the table, floor and goal markers. Recompiles the model per tool. |
| `tool_gen.py` | Random tool: handle (cylinder/capsule/box) + head (box/cylinder/sphere/ellipsoid/capsule) in a T, L or inline layout |
| `goals.py` | Goal sampling from a precomputed reach map, plus the segment-based pass-through test |
| `rewards.py` | Dense and terminal reward terms |
| `env.py` | `PoseVelocityEnv` (Gymnasium) |
| `vec_env.py` | Subprocess vector env with same-step autoreset that also returns privileged critic features |
| `ppo.py` | Asymmetric actor-critic, observation normalizers, PPO update, GAE |
| `train.py` | Training: rollouts, curriculum, TensorBoard, checkpoints |
| `evaluate.py` | Evaluation: viewer, video, success stats |

## How an episode works

- **Goal:** a point `p_g`, a velocity `v_g` and a head axis `a_g`. By default `a_g` points along `v_g`, like a strike. All three are sampled inside the arm's reach. The point slightly before the goal along the motion (lead-in) and slightly after it (follow-through) must also be reachable.
- **Success:** checked once per control step against the head's path since the previous step, so fast passes aren't missed. While the tool is held, all three must hold:
  - the closest approach to `p_g` is within `eps_p`;
  - `|v_head - v_g| < eps_v_frac · |v_g|`, a fraction of the goal speed, so a stationary head never qualifies;
  - the angle between the head axis and `a_g` is below `eps_theta`.

  The curriculum tightens these from 8 cm / 60% of goal speed / 60° to 3 cm / 25% of goal speed / 20°.
- **Drop:** the tool had been grasped and then either lost all hand contact for 0.2 s or touched the floor. It also counts as a drop if the tool is knocked off the table. A drop ends the episode early.
- **Time limit:** 10 s (500 control steps at 50 Hz).
- **After an episode:**
  - The arm is **never** re-homed.
  - **Soft reset:** if the tool is still held, the whole physics state is kept and only the goal changes.
  - **Respawn:** otherwise, a new tool is generated and placed on the table (or in the hand, early in the curriculum), and the arm stays where it was.
  - The arm returns to its home pose only on the first reset, on `reset(seed=...)`, or after the physics diverges.

## Model inputs and outputs

- **Action (22 values in [-1, 1]):** changes to the joint-position targets.
  - Arm: 6 joints, up to 0.03 rad per step.
  - Hand: 16 joints, up to 0.08 rad per step.
- **Actor observation (about 160 values):**
  - arm and hand joint positions and velocities;
  - the previous action;
  - palm pose and fingertip positions;
  - fingertip contact flags;
  - tool head pose and velocity;
  - shape features of the tool;
  - the goal, and the position, velocity and axis errors to it;
  - grasp flags and time remaining.
- **Critic-only inputs** (`info["priv"]`): tool mass, center of mass and friction; contact forces; best distance so far; curriculum level.
- **Networks:**
  - Actor: MLP 512-256-128 (ELU) with a Gaussian output.
  - Critic: MLP 512-512-256.
  - Both use running observation normalization.
- **Reward:**
  - Reaching the handle, fingertip contact and lifting are combined into a grasp potential Φ, rewarded as potential-based shaping `γΦ' − Φ`. Progress toward a grasp pays once; just holding the tool pays nothing; dropping it costs the same amount back.
  - Goal attraction while holding is also potential-based: Φ_goal = wide position term + tight `exp(−D)` term. Approaching the goal pays, hovering near it pays nothing. Closest-approach progress pays for each new best D.
  - Episode boundaries after a soft reset (success, or a time-out with the tool still held) don't cut the PPO return. The world carries on with a new goal, so finishing a goal never forfeits future reward.
  - Small penalties on actions and on the robot touching the table.
  - Terminal terms: success +50 (plus a bonus for finishing early), drop -20, and `10·exp(-D_min)` for how close the attempt got.
  - The weights are in `RewardCfg`.

`PoseVelocityEnv` is a standard Gymnasium env, so SB3 also works as a quick baseline: `PPO("MlpPolicy", PoseVelocityEnv())`. The SB3 baseline uses only the actor observation.

## Troubleshooting

- **`AttributeError` mentioning `MjSpec` / `attach` / `copy`:** your MuJoCo is too old. Run `pip install -U "mujoco>=3.3"`.
- **`ImportError: robot_descriptions`:** install it, or set the two `POSEVEL_*_XML` variables shown above.
- **Hand mount:** by default (`auto_align_hand=True`) the fingers are aligned with the flange axis and the palm sits flush on the flange. To change which way the palm faces, set `hand_mount_euler = (0, 0, angle)`, which spins the hand about the wrist axis. Use `hand_mount_pos` for small offsets. Re-run `python scene.py --view` to check.
- **Viewer does not open under WSL:** WSLg is needed (`echo $DISPLAY` should print something). Use `--video` as a headless alternative.
- **Headless video rendering fails:** run `export MUJOCO_GL=egl` (or `osmesa`) first.
