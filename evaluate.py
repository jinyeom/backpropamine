from time import sleep
import numpy as np
import torch
from torch import distributions

from modules import RecurrentBackpropamineAgent
from envs import MiconiMaze
from utils import NumpyTube


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MiconiMaze(16)
    nptube = NumpyTube()

    agent = RecurrentBackpropamineAgent(env.obs_shape, action_size=env.action_size).to(device)
    model_state_dict = torch.load("artifacts/MiconiMaze-027edb82a64c44a4a6256579729ed953/model_final.pth")
    agent.load_state_dict(model_state_dict)

    print(agent)

    obs = env.reset().to(device)
    h = hebb = None
    prev_action = torch.zeros(16, env.action_size).to(device)
    prev_reward = torch.zeros(16, 1).to(device)

    for i in range(200):
        nptube.imshow(env.render())
        with torch.no_grad():
            action_probs, value_pred, m, h, hebb = agent(obs, prev_action, prev_reward, h, hebb)
        pi = distributions.OneHotCategorical(probs=action_probs)
        action_one_hot = pi.sample()
        action = torch.argmax(action_one_hot, dim=1).cpu()

        obs, reward = env.step(action)
        obs = obs.to(device)
        reward = reward.to(device)

        prev_action = action_one_hot
        prev_reward = reward

        sleep(5e-2)

    nptube.close()
