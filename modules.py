import math

import torch
from torch import nn
from torch.nn import functional as F


class ModulatedPlasticDense(nn.Module):
    def __init__(self, in_features, out_features, clip=2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.clip = clip

        self.weight = nn.Parameter(torch.Tensor(in_features + 1, out_features))
        self.alpha = nn.Parameter(torch.Tensor(in_features + 1, out_features))
        self.ln = nn.LayerNorm(out_features)

        self.modulator = nn.Linear(out_features, 1)
        self.modfanout = nn.Linear(1, out_features)  # per-neuron

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.001)
        nn.init.normal_(self.alpha, std=0.001)

    def forward(self, x, hebb):
        x = F.pad(x, (0, 1), "constant", 1.0)  # bias

        weight = self.weight + self.alpha * hebb
        y = torch.tanh(self.ln((x.unsqueeze(1) @ weight).squeeze(1)))

        # neuromodulated plasticity update
        m = torch.tanh(self.modulator(y))
        eta = self.modfanout(m.unsqueeze(2))
        delta = eta * (x.unsqueeze(2) @ y.unsqueeze(1))
        hebb = torch.clamp(hebb + delta, min=-self.clip, max=self.clip)

        return y, m, hebb


class ModulatedPlasticRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, clip=2.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.clip = clip

        self.fc = nn.Linear(input_size, hidden_size)

        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.alpha = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.ln = nn.LayerNorm(hidden_size)

        self.modulator = nn.Linear(hidden_size, 1)
        self.modfanout = nn.Linear(1, hidden_size)  # per-neuron

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.001)
        nn.init.normal_(self.alpha, std=0.001)

    def forward(self, x, h_pre, hebb):
        weight = self.weight + self.alpha * hebb
        h_post = torch.tanh(self.ln(self.fc(x) + (h_pre.unsqueeze(1) @ weight).squeeze(1)))

        # neuromodulated plasticity update
        m = torch.tanh(self.modulator(h_post))
        eta = self.modfanout(m.unsqueeze(2))
        delta = eta * (h_pre.unsqueeze(2) @ h_post.unsqueeze(1))
        hebb = torch.clamp(hebb + delta, min=-self.clip, max=self.clip)

        return h_post, m, hebb


class ForwardBackpropamineAgent(nn.Module):
    def __init__(self, obs_shape, feature_size=64, hidden_size=128, action_size=4):
        super().__init__()
        self.obs_shape = obs_shape
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        C, H, W = obs_shape
        self.enc = nn.Sequential(
            nn.Conv2d(C, 16, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * (H - 2) * (W - 2), feature_size),
            nn.ReLU(inplace=True),
        )

        self.hidden1 = nn.Linear(feature_size + action_size + 1, hidden_size)
        self.hidden2 = ModulatedPlasticDense(hidden_size, hidden_size)

        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs, prev_action, prev_reward, h=None, hebb=None):
        batch_size = obs.size(0)
        device = obs.device

        obs = self.enc(obs)
        x = torch.cat([obs, prev_action, prev_reward], dim=1)

        x = torch.tanh(self.hidden1(x))

        if hebb is None:
            hebb = torch.zeros(batch_size, *self.hidden2.weight.shape).to(device)
        x, m, hebb = self.hidden2(x, hebb)

        action_probs = F.softmax(self.actor(x), dim=1)
        value_pred = self.critic(x)

        return action_probs, value_pred, m, None, hebb


class RecurrentBackpropamineAgent(nn.Module):
    def __init__(self, obs_shape, feature_size=64, hidden_size=128, action_size=4):
        super().__init__()
        self.obs_shape = obs_shape
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        C, H, W = obs_shape
        self.enc = nn.Sequential(
            nn.Conv2d(C, 16, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * (H - 2) * (W - 2), feature_size),
            nn.ReLU(inplace=True),
        )

        self.prnn = ModulatedPlasticRNNCell(feature_size + action_size + 1, hidden_size)

        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs, prev_action, prev_reward, h=None, hebb=None):
        batch_size = obs.size(0)
        device = obs.device

        x = torch.cat([self.enc(obs), prev_action, prev_reward], dim=1)

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size).to(device)
        if hebb is None:
            hebb = torch.zeros(batch_size, *self.prnn.weight.shape).to(device)
        h, m, hebb = self.prnn(x, h, hebb)

        action_probs = F.softmax(self.actor(h), dim=1)
        value_pred = self.critic(h)

        return action_probs, value_pred, m, h, hebb
