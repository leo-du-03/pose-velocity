import numpy as np
import torch

from config import PPOCfg
from ppo import PPO, ActorCritic


def test_critic_warmup_leaves_actor_untouched():
    cfg = PPOCfg(minibatches=2, epochs=2)
    agent = ActorCritic(8, 3, 2, cfg)
    ppo = PPO(agent, cfg)
    n = 64
    b = {"obs": torch.randn(n, 8), "priv": torch.randn(n, 3), "actions": torch.randn(n, 2) * 0.1,
         "logp": torch.zeros(n), "values": torch.zeros(n), "adv": torch.randn(n), "returns": torch.randn(n)}
    with torch.no_grad():
        b["logp"] = agent.dist(b["obs"]).log_prob(b["actions"]).sum(-1)
    actor0 = [p.detach().clone() for p in ppo.actor_params]
    critic0 = [p.detach().clone() for p in ppo.critic_params]
    ppo.update(b, update_actor=False)
    assert all(torch.equal(a, p) for a, p in zip(actor0, ppo.actor_params))
    assert any(not torch.equal(c, p) for c, p in zip(critic0, ppo.critic_params))


def test_env_oracle_action_matches_oracle(env):
    from oracle import Oracle

    ref = Oracle()
    for _ in range(40):
        a = env.oracle_action()
        np.testing.assert_allclose(a, ref(env), atol=1e-6)
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset()
