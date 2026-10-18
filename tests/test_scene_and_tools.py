import numpy as np

from config import SceneCfg, ToolCfg
from scene import SceneBuilder
from tool_gen import FEATURE_DIM, sample_tool


def test_robot_discovery():
    b = SceneBuilder(SceneCfg())
    ri = b.info
    assert len(ri.arm_joints) == 6, ri.arm_joints
    assert len(ri.hand_joints) == 16, ri.hand_joints
    assert len(ri.fingertip_bodies) == 4, ri.fingertip_bodies
    assert np.linalg.norm(ri.palm_home[:2]) > 0.15, "palm at home should be in front of the base"


def test_tools_compile_with_mass_bounds():
    b = SceneBuilder(SceneCfg())
    rng = np.random.default_rng(0)
    tcfg = ToolCfg()
    for _ in range(50):
        tool = sample_tool(rng, tcfg)
        assert tool.features().shape == (FEATURE_DIM,)
        m = b.compile(tool)
        tid = m.body("tool").id
        assert tcfg.mass[0] * 0.9 <= m.body_mass[tid] <= tcfg.mass[1] * 1.1, m.body_mass[tid]
        assert np.isclose(np.linalg.norm(tool.head_axis), 1.0)


def test_weld_compiles_with_grip_constraint():
    b = SceneBuilder(SceneCfg())
    tool = sample_tool(np.random.default_rng(0), ToolCfg())
    m = b.compile(tool, weld=True)
    assert m.neq == 1 and m.eq(0).name == "grip"
    np.testing.assert_allclose(m.eq_data[0, :3], 0.0)  # the anchor must be zero (MjSpec defaults it to 0 1 0)
    assert b.compile(tool, weld=False).neq == 0


def test_grasp_scene_starts_floating():
    """Grasp grip: the grip weld and the holder start inactive, the tool is gravity compensated, and only the hand
    lacks collision bit 2 (so the weld can switch off hand-tool contacts alone)."""
    b = SceneBuilder(SceneCfg())
    tool = sample_tool(np.random.default_rng(0), ToolCfg())
    m = b.compile(tool, weld=True, grasp=True)
    assert [m.eq(i).name for i in range(m.neq)] == ["grip", "hold"]
    assert not m.eq_active0.any()
    np.testing.assert_allclose(m.eq_data[:, :3], 0.0)
    assert m.body_gravcomp[m.body("tool").id] == 1.0
    for g in range(m.ngeom):
        if m.geom_contype[g]:
            in_hand = m.body(int(m.geom_bodyid[g])).name.startswith("hand/")
            assert bool(m.geom_contype[g] & 2) != in_hand


def test_tools_rest_on_table(physical_env):
    """Respawned tools should settle on the table without flying off."""
    env = physical_env
    env.set_curriculum(1.0)  # no in-hand starts
    for _ in range(15):
        env.reset(options={"kind": "respawn"})
        env._settle(1.0)
        z = env.data.xpos[env.tool_id][2]
        v = np.linalg.norm(env.data.qvel[env.tool_dadr:env.tool_dadr + 3])
        assert env.rinfo.table_top - 0.01 < z < env.rinfo.table_top + 0.15, z
        assert v < 0.1, v
