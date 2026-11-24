import numpy as np
import pytest
import torch

from dog2real.rl.networks import ActorCritic
from dog2real.rl.normalizer import EmpiricalNormalization
from dog2real.rl.ppo import PPO, gaussian_kl
from dog2real.rl.runner import OnPolicyRunner
from dog2real.rl.storage import compute_gae
from dog2real.rl.toy_env import PointMassEnv


def test_gae_matches_hand_computation():
    gamma, lam = 0.9, 0.8
    r = torch.tensor([[1.0], [2.0], [3.0]])
    v = torch.tensor([[0.5], [1.0], [1.5]])
    d = torch.tensor([[0.0], [1.0], [0.0]])  # episode ends after t=1
    last_v = torch.tensor([2.0])
    adv, ret = compute_gae(r, v, d, last_v, gamma, lam)
    # by hand:
    delta2 = 3.0 + 0.9 * 2.0 - 1.5          # 3.3
    a2 = delta2
    delta1 = 2.0 + 0.0 - 1.0                # terminal: no bootstrap -> 1.0
    a1 = delta1                              # (1 - d1) kills the recursion
    delta0 = 1.0 + 0.9 * 1.0 - 0.5           # 1.4
    a0 = delta0 + 0.9 * 0.8 * a1             # 2.12
    np.testing.assert_allclose(adv.squeeze(-1).numpy(), [a0, a1, a2], rtol=1e-6)
    np.testing.assert_allclose(ret.squeeze(-1).numpy(), np.array([a0, a1, a2]) + [0.5, 1.0, 1.5], rtol=1e-6)


def test_gae_lambda_one_equals_discounted_return_minus_value():
    torch.manual_seed(0)
    T, gamma = 6, 0.95
    r = torch.randn(T, 1)
    v = torch.randn(T, 1)
    last_v = torch.randn(1)
    adv, _ = compute_gae(r, v, torch.zeros(T, 1), last_v, gamma, 1.0)
    G = last_v.clone()
    for t in reversed(range(T)):
        G = r[t] + gamma * G
        assert adv[t].item() == pytest.approx((G - v[t]).item(), rel=1e-5)


def test_normalizer_matches_numpy():
    torch.manual_seed(0)
    norm = EmpiricalNormalization(3)
    data = [torch.randn(50, 3) * torch.tensor([1.0, 5.0, 0.1]) + torch.tensor([2.0, -1.0, 0.0]) for _ in range(4)]
    for x in data:
        norm.update(x)
    allx = torch.cat(data).numpy()
    np.testing.assert_allclose(norm.mean.numpy(), allx.mean(0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(norm.var.numpy(), allx.var(0), rtol=1e-4)
    y = norm(torch.cat(data))
    assert y.abs().max() <= 5.0


def test_gaussian_kl_zero_for_identical_and_positive_otherwise():
    mu, s = torch.zeros(4, 2), torch.ones(4, 2)
    assert torch.allclose(gaussian_kl(mu, s, mu, s), torch.zeros(4), atol=1e-6)
    assert torch.all(gaussian_kl(mu, s, mu + 0.5, s * 1.2) > 0)
    # closed form for 1-D: log(s2/s1) + (s1^2 + (m1-m2)^2) / (2 s2^2) - 1/2
    kl = gaussian_kl(torch.tensor([[0.0]]), torch.tensor([[1.0]]), torch.tensor([[1.0]]), torch.tensor([[2.0]]))
    assert kl.item() == pytest.approx(np.log(2) + 2 / 8 - 0.5, rel=1e-5)


def test_ppo_update_decreases_loss_on_fixed_batch():
    torch.manual_seed(0)
    policy = ActorCritic(4, 4, 2, [32], [32])
    alg = PPO(policy, num_epochs=1, num_minibatches=1, learning_rate=3e-3, schedule="fixed", entropy_coef=0.0)
    alg.init_storage(16, 8, 4, 4, 2)
    obs = torch.randn(16, 8, 4)
    for t in range(16):
        alg.act(obs[t], obs[t])
        # reward favours action[0] > 0, and has a learnable value structure
        rew = alg._pending[2][:, 0].clamp(-1, 1) + obs[t, :, 0]
        alg.process_env_step(rew, torch.zeros(8), torch.zeros(8))
    alg.compute_returns(obs[-1])
    st = alg.storage
    snapshot = [x.clone() for x in (st.obs, st.critic_obs, st.actions, st.log_probs, st.mu, st.sigma, st.values,
                                    st.advantages, st.returns)]

    def loss_on_batch():
        with torch.no_grad():
            o, c, a, lp, mu, sg, v, adv, ret = [x.reshape(128, *x.shape[2:]) for x in snapshot]
            dist = policy.distribution(o)
            ratio = torch.exp(dist.log_prob(a).sum(-1) - lp)
            advn = (adv - adv.mean()) / (adv.std() + 1e-8)
            surr = -torch.min(ratio * advn, ratio.clamp(0.8, 1.2) * advn).mean()
            vl = (policy.evaluate(c) - ret).pow(2).mean()
            return (surr + vl).item()

    before = loss_on_batch()
    for _ in range(10):
        st.obs, st.critic_obs, st.actions, st.log_probs, st.mu, st.sigma, st.values, st.advantages, st.returns = \
            [x.clone() for x in snapshot]
        alg.update()
    after = loss_on_batch()
    assert after < before


def test_ppo_solves_point_mass_task():
    torch.manual_seed(0)
    env = PointMassEnv(num_envs=32, episode_length=30, seed=0)
    cfg = {"seed": 0,
           "policy": {"actor_hidden": [32, 32], "critic_hidden": [32, 32], "init_noise_std": 0.5,
                      "asymmetric_critic": False},
           "ppo": {"steps_per_env": 30, "num_epochs": 5, "num_minibatches": 4, "learning_rate": 3e-3,
                   "entropy_coef": 0.0, "gamma": 0.95},
           "runner": {"tensorboard": False}}
    runner = OnPolicyRunner(env, cfg)
    hist = runner.learn(40, verbose=False)
    first = hist[0]["train/mean_step_reward"]
    last = np.mean([h["train/mean_step_reward"] for h in hist[-5:]])
    assert first < -0.4          # random-ish initial policy
    assert last > -0.15          # learned to drive the point to the origin
    # deterministic policy check
    with torch.no_grad():
        x = torch.tensor([[0.8, -0.6]])
        a = runner.policy.act_inference(runner.obs_norm(x))
    assert a[0, 0] < 0 and a[0, 1] > 0  # pushes towards the origin


def test_checkpoint_roundtrip(tmp_path):
    env = PointMassEnv(num_envs=8, seed=0)
    cfg = {"policy": {"actor_hidden": [16], "critic_hidden": [16], "asymmetric_critic": False},
           "ppo": {"steps_per_env": 8}, "runner": {"tensorboard": False, "save_interval": 1}}
    r1 = OnPolicyRunner(env, cfg, log_dir=tmp_path)
    r1.learn(2, verbose=False)
    assert (tmp_path / "model_last.pt").exists() and (tmp_path / "progress.csv").exists()
    r2 = OnPolicyRunner(env, cfg)
    r2.load(tmp_path / "model_last.pt")
    assert r2.iteration == 2
    x = torch.randn(5, 2)
    torch.testing.assert_close(r1.policy.act_inference(r1.obs_norm(x)), r2.policy.act_inference(r2.obs_norm(x)))
    lines = (tmp_path / "progress.csv").read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 iterations
