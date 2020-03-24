import torch
from torch import distributions, optim
from torch.nn.utils import clip_grad_norm_
from matplotlib import pyplot as plt

from modules import ForwardBackpropamineAgent, RecurrentBackpropamineAgent
from envs import make_env
from utils import Experiment


def main(args):
    experiment = Experiment(args)
    experiment.show_args()
    experiment.export_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(args.env_id, args.batch_size, seed=args.seed)

    if args.recurrent:
        agent = RecurrentBackpropamineAgent(
            env.obs_shape,
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            action_size=env.action_size,
        ).to(device)
    else:
        agent = ForwardBackpropamineAgent(
            env.obs_shape,
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            action_size=env.action_size,
        ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_gamma
    )

    for lifetime in range(args.num_lifetimes):
        obs = env.reset().to(device)

        h = hebb = None

        # initialize previous actions and rewards
        prev_action = torch.zeros(args.batch_size, env.action_size).to(device)
        prev_reward = torch.zeros(args.batch_size, 1).to(device)

        lifetime_reward = []
        lifetime_action_log_prob = []
        lifetime_entropy = []
        lifetime_value_pred = []

        for step in range(args.lifetime_length):
            action_probs, value_pred, m, h, hebb = agent(obs, prev_action, prev_reward, h, hebb)
            pi = distributions.OneHotCategorical(probs=action_probs)
            action_one_hot = pi.sample()
            action_log_prob = pi.log_prob(action_one_hot).unsqueeze(1)
            entropy = pi.entropy().unsqueeze(1)

            action = torch.argmax(action_one_hot, dim=1).cpu()
            obs, reward = env.step(action)
            obs = obs.to(device)
            reward = reward.to(device)

            prev_action = action_one_hot
            prev_reward = reward

            lifetime_reward.append(reward)
            lifetime_action_log_prob.append(action_log_prob)
            lifetime_entropy.append(entropy)
            lifetime_value_pred.append(value_pred)

        # lifetime over! compute losses and train the agent

        policy_loss = 0.0
        value_loss = 0.0

        gae = 0.0  # generalized advantage estimation
        ret = 0.0  # return / utility

        lifetime_value_pred.append(0.0)  # assuming end of "episode"

        for t in reversed(range(args.lifetime_length)):
            reward = lifetime_reward[t]
            value_next = lifetime_value_pred[t + 1]
            value = lifetime_value_pred[t]
            action_log_prob = lifetime_action_log_prob[t]
            entropy = lifetime_entropy[t]

            ret = args.gamma * ret + reward
            td_err = reward + args.gamma * value_next - value
            gae = args.gamma * args.lam * gae + td_err

            policy_loss -= action_log_prob * gae.detach() + args.entropy_coef * entropy
            value_loss += 0.5 * (ret - value) ** 2

        policy_loss = policy_loss.mean(dim=0) / args.lifetime_length
        value_loss = value_loss.mean(dim=0) / args.lifetime_length

        loss = policy_loss + args.value_coef * value_loss
        mean_reward = sum(lifetime_reward).mean(dim=0).item()

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if (lifetime + 1) % args.log_interval == 0:
            experiment.log(
                lifetime,
                policy_loss=policy_loss.item(),
                value_loss=value_loss.item(),
                total_loss=loss.item(),
                mean_reward=mean_reward,
            )

        if (lifetime + 1) % args.checkpoint_interval == 0:
            experiment.checkpoint(lifetime, agent, optimizer, loss)

        if (lifetime + 1) % args.update_best_interval == 0:
            experiment.update_best(agent, mean_reward)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-path", type=str, default='/tmp')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-id", type=str, default="MiconiMaze")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-lifetimes", type=int, default=40000)
    parser.add_argument("--lifetime-length", type=int, default=200)
    parser.add_argument("--recurrent", action="store_true", default=False)
    parser.add_argument("--feature-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-decay-step-size", type=int, default=10000)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=4.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--update-best-interval", type=int, default=100)
    args = parser.parse_args()

    main(args)
