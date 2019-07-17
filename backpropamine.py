import os
import json
from time import ctime
from pprint import pprint

import torch
from torch import nn, optim, distributions
from torch.nn import functional as F
from torchvision.utils import make_grid
import pyglet
from pyglet import gl
from pyglet.window import Window
import numpy as np
from tqdm import tqdm


# learning to reinforcement learn (L2RL) agent (Wang et al., 2017) with a simple recurrent layer
class SimpleL2RL(nn.Module):
  def __init__(self, obs_dim, act_dim, hid_dim):
    super(SimpleL2RL, self).__init__()
    self.obs_dim = obs_dim
    self.act_dim = act_dim
    self.hid_dim = hid_dim

    self.i2h = nn.Linear(obs_dim, hid_dim)
    self.W_hh = nn.Parameter(0.001 * torch.rand(hid_dim, hid_dim))  # parameter of interest!
    self.actor = nn.Linear(hid_dim, act_dim)  # from recurrent to action probabilities
    self.critic = nn.Linear(hid_dim, 1)  # from recurrent to value prediction

  def forward(self, obs, h):
    h = torch.tanh(self.i2h(obs) + torch.mm(h, self.W_hh))
    act_prob = F.softmax(self.actor(h), dim=-1)
    value_pred = self.critic(h)
    return act_prob, value_pred, h


# learning to reinforcement learn (L2RL) agent (Wang et al., 2017) with a long short-term memory
# NOTE: this one is for checking if the L2RL algorithm depends on LSTMs; if it does, neuromodulation
# would probably look much more attractive, as it may be a more general approach
#
# additionally, if this model can either match or outperform neuromodulation models, we'll see how
# versatile LSTMs can be, yet again
class LstmL2RL(nn.Module):
  def __init__(self, obs_dim, act_dim, hid_dim):
    super(LstmL2RL, self).__init__()
    self.obs_dim = obs_dim
    self.act_dim = act_dim
    self.hid_dim = hid_dim

    self.lstm = nn.LSTMCell(obs_dim, hid_dim)
    self.actor = nn.Linear(hid_dim, act_dim)  # from recurrent to action probabilities
    self.critic = nn.Linear(hid_dim, 1)  # from recurrent to value prediction

  def forward(self, obs, hidden):
    h, c = self.lstm(obs, hidden)
    act_prob = F.softmax(self.actor(h), dim=-1)
    value_pred = self.critic(h)
    return act_prob, value_pred, (h, c)


# agent with simple neuromodulation from https://github.com/uber-research/backpropamine/blob/master/maze/maze.py
class SimpleNeuromodulation(nn.Module):
  def __init__(self, obs_dim, act_dim, hid_dim): 
    super(SimpleNeuromodulation, self).__init__()
    self.obs_dim = obs_dim
    self.act_dim = act_dim
    self.hid_dim = hid_dim

    self.i2h = nn.Linear(obs_dim, hid_dim)
    self.W_hh = nn.Parameter(0.001 * torch.rand(hid_dim, hid_dim))  # parameter of interest!
    self.modfanout = nn.Linear(1, hid_dim)  # fan out modulator output for each neuron
    self.alpha = nn.Parameter(0.001 * torch.rand(hid_dim, hid_dim))  # one alpha coefficient per recurrent connection
    # self.alpha = torch.nn.Parameter(0.0001 * torch.rand(1, 1, hid_dim))  # one alpha coefficient per neuron
    # self.alpha = torch.nn.Parameter(0.0001 * torch.ones(1))  # single alpha coefficient for the whole network

    self.actor = nn.Linear(hid_dim, act_dim)  # from recurrent to action probabilities
    self.critic = nn.Linear(hid_dim, 1)  # from recurrent to value prediction
    self.modulator = nn.Linear(hid_dim, 1)  # from the recurrent to neuromodulator output
   
  def mod_W_hh(self, hebb):
    return self.W_hh + self.alpha * hebb

  def forward(self, obs, h_pre, hebb):
    h_post = torch.tanh(self.i2h(obs) + torch.bmm(h_pre.unsqueeze(1), self.mod_W_hh(hebb)).squeeze(1))
    act_prob = F.softmax(self.actor(h_post), dim=-1)
    value_pred = self.critic(h_post)
    m = torch.tanh(self.modulator(h_post))
    delta_hebb = torch.bmm(h_pre.unsqueeze(2), h_post.unsqueeze(1))
    hebb = torch.clamp(hebb + self.modfanout(m.unsqueeze(2)) * delta_hebb, min=-2, max=2)
    return act_prob, value_pred, m, h_post, hebb


class SimpleNeuromodulationLSTM(nn.Module):
  def __init__(self, obs_dim, act_dim, hid_dim):
    super(SimpleNeuromodulationLSTM, self).__init__()
    self.obs_dim = obs_dim
    self.act_dim = act_dim
    self.hid_dim = hid_dim

    self.i2h = nn.Linear(obs_dim, hid_dim * 4)  # for each gate
    self.gate_W_hh = nn.Parameter(0.001 * torch.rand(hid_dim, hid_dim * 3))  # for each of forget, input, output gates
    self.cell_W_hh = nn.Parameter(0.001 * torch.rand(hid_dim, hid_dim))  # parameter of interest!
    self.modfanout = nn.Linear(1, hid_dim)  # fan out modulator output for each neuron
    self.alpha = nn.Parameter(0.001 * torch.rand(hid_dim, hid_dim))  # one alpha coefficient per recurrent connection

    self.actor = nn.Linear(hid_dim, act_dim)  # from recurrent to action probabilities
    self.critic = nn.Linear(hid_dim, 1)  # from recurrent to value prediction
    self.modulator = nn.Linear(hid_dim, 1)  # from the recurrent to neuromodulator output

  def mod_W_hh(self, hebb):
    return self.cell_W_hh + self.alpha * hebb

  def forward(self, obs, hidden, hebb):
    h_pre, c = hidden
    gates = self.i2h(obs) + torch.cat([
      torch.mm(h_pre, self.gate_W_hh),
      torch.bmm(h_pre.unsqueeze(1), self.mod_W_hh(hebb)).squeeze(1)
    ], dim=-1)
    f = torch.sigmoid(gates[:, :self.hid_dim])
    i = torch.sigmoid(gates[:, self.hid_dim:self.hid_dim*2])
    o = torch.sigmoid(gates[:, self.hid_dim*2:self.hid_dim*3])
    c = f * c + i * torch.tanh(gates[:, self.hid_dim*3:])
    h_post = o * torch.tanh(c)
    act_prob = F.softmax(self.actor(h_post), dim=-1)
    value_pred = self.critic(h_post)
    m = torch.tanh(self.modulator(h_post))
    delta_hebb = torch.bmm(h_pre.unsqueeze(2), h_post.unsqueeze(1))
    hebb = torch.clamp(hebb + self.modfanout(m.unsqueeze(2)) * delta_hebb, min=-2, max=2)
    return act_prob, value_pred, m, (h_post, c), hebb


# agent with retroactive neuromodulation and eligibility traces
class RetroactiveNeuromodulation(nn.Module):
  def __init__(self, obs_dim, act_dim, hid_dim): 
    super(RetroactiveNeuromodulation, self).__init__()
    self.obs_dim = obs_dim
    self.act_dim = act_dim
    self.hid_dim = hid_dim

    self.i2h = nn.Linear(obs_dim, hid_dim)
    self.W_hh = nn.Parameter(0.001 * torch.rand(hid_dim, hid_dim))  # parameter of interest!
    self.modfanout = nn.Linear(1, hid_dim)  # fan out modulator output for each neuron
    self.alpha = nn.Parameter(0.001 * torch.rand(hid_dim, hid_dim))  # one alpha coefficient per recurrent connection
    self.eta = nn.Parameter(0.001 * torch.rand(1))  # trainable decay factor for the eligibility trace
    # self.alpha = torch.nn.Parameter(0.0001 * torch.rand(1, 1, hid_dim))  # one alpha coefficient per neuron
    # self.alpha = torch.nn.Parameter(0.0001 * torch.ones(1))  # single alpha coefficient for the whole network

    self.actor = nn.Linear(hid_dim, act_dim)  # from recurrent to action probabilities
    self.critic = nn.Linear(hid_dim, 1)  # from recurrent to value prediction
    self.modulator = nn.Linear(hid_dim, 1)  # from the recurrent to neuromodulator output
   
  def mod_W_hh(self, hebb):
    return self.W_hh + self.alpha * hebb

  def forward(self, obs, h_pre, hebb, trace):
    h_post = torch.tanh(self.i2h(obs) + torch.bmm(h_pre.unsqueeze(1), self.mod_W_hh(hebb)).squeeze(1))
    act_prob = F.softmax(self.actor(h_post), dim=-1)
    value_pred = self.critic(h_post)
    m = torch.tanh(self.modulator(h_post))
    hebb = torch.clamp(hebb + self.modfanout(m.unsqueeze(2)) * trace, min=-2, max=2)
    delta_hebb = torch.bmm(h_pre.unsqueeze(2), h_post.unsqueeze(1))
    trace = (1 - self.eta) * trace + self.eta * delta_hebb
    return act_prob, value_pred, m, h_post, hebb, trace


class ImageViewer:
  def __init__(self, display=None, maxwidth=500):
    self.window = None
    self.isopen = False
    self.display = display

  def __del__(self):
    self.close()

  def imshow(self, arr, caption):
    if self.window is None:
      height, width, _ = arr.shape
      self.window = Window(width=width, height=height, display=self.display, vsync=False, resizable=True)
      self.width = width
      self.height = height
      self.isopen = True
    assert len(arr.shape) == 3
    height, width, _ = arr.shape
    image = pyglet.image.ImageData(width, height, 'RGB', arr.tobytes(), pitch=-3*width)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    texture = image.get_texture()
    texture.width = self.width
    texture.height = self.height
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    texture.blit(0, 0)
    self.window.flip()
    self.window.set_caption(caption)

  def close(self):
    if self.isopen:
      self.window.close()
      self.isopen = False


class MazeNavigationTask:
  GOAL_REW = 10.0  # reward for reaching the goal
  WALL_PEN = -0.1  # penalty for running into a wall

  def __init__(self, lab_size, batch_size):
    self.lab_size = lab_size
    self.batch_size = batch_size
    self.obs_dim = 14  # receptive field (9), one-hot action (4), previous reward (1)
    self.act_dim = 4  # up, down, left, right

    self.lab = np.ones((lab_size, lab_size))
    self.lab[1:lab_size-1, 1:lab_size-1].fill(0)
    for i in range(1, lab_size - 1):
      for j in range(1, lab_size - 1):
        if i % 2 == 0 and j % 2 == 0:
          self.lab[i, j] = 1
    # Not strictly necessary, but cleaner since we start the agent at the
    # center for each episode; may help localization in some maze sizes
    # (including 13 and 9, but not 11) by introducing a detectable irregularity
    # in the center:
    ctr = lab_size // 2
    self.lab[ctr, ctr] = 0
    self.viewer = ImageViewer()  # for rendering maze lab

  def reset(self, device):
    # reset values that go into observation
    ctr = self.lab_size//2
    self.agent_pos = np.full((self.batch_size, 2), ctr)
    self.prev_action = np.zeros((self.batch_size, self.act_dim))
    self.prev_rews = np.zeros((self.batch_size, 1), dtype=np.float32)

    # reset goals to random positions
    self.goals = np.zeros((self.batch_size, 2))
    for batch in range(self.batch_size):
      goal_y, goal_x = np.random.randint(1, self.lab_size-1, size=2)
      while self.lab[goal_y, goal_x] == 1 or (goal_y == ctr and goal_x == ctr):
        goal_y, goal_x = np.random.randint(1, self.lab_size-1, size=2)
      self.goals[batch] = (goal_y, goal_x)

    # construct the initial observation batch
    obs = np.zeros((self.batch_size, self.obs_dim), dtype=np.float32)
    for batch in range(self.batch_size):
      rf = self.lab[self.agent_pos[batch][0]-1:self.agent_pos[batch][0]+2, 
                    self.agent_pos[batch][1]-1:self.agent_pos[batch][1]+2]
      obs[batch] = np.concatenate([rf.ravel(), self.prev_action[batch], self.prev_rews[batch]])

    return torch.from_numpy(obs).to(device)

  def paint_lab(self, scale=8):
    height, width = self.lab.shape
    def sqrtish(n):
      sqrtish = np.ceil(n**0.5)
      while n % sqrtish != 0:
        sqrtish -= 1
      a = int(sqrtish)
      b = int(n//sqrtish)
      assert n == a * b
      return a, b
    nrows, ncols = sqrtish(len(self.agent_pos))
    mat = []
    for i in range(nrows):
      row = []
      for j in range(ncols):
        batch = i * ncols + j
        canvas = np.zeros((height, width, 3))
        for r in range(height):
          for c in range(width):
            if self.lab[r, c] == 1:
              canvas[r, c] = (0, 0, 0)  # walls are black
            elif self.lab[r, c] == 0:
              canvas[r, c] = (255, 255, 255)  # floors are white
        goal = tuple(self.goals[batch].astype(np.uint8))
        pos = tuple(self.agent_pos[batch].astype(np.uint8))
        canvas[goal] = (255, 0, 0)  # goals is red
        canvas[pos] = (0, 128, 255)  # agents are blue
        row.append(canvas)
      mat.append(np.hstack(row))
    art = np.vstack(mat)
    return np.kron(art, np.ones((scale, scale, 1))).astype(np.uint8)

  def render(self):
    self.viewer.imshow(self.paint_lab(), caption=f'maze navigation')

  def step(self, action):  # takes in one-hot encoded action batch
    self.prev_action = action.cpu().numpy()
    rew = np.zeros(self.batch_size, dtype=np.float32)
    for batch in range(self.batch_size):
      act_idx = np.argmax(self.prev_action[batch])  # action index
      agent_y, agent_x = self.agent_pos[batch]
      if act_idx == 0:  # move up
        agent_y -= 1
      elif act_idx == 1:  # move down
        agent_y += 1
      elif act_idx == 2:  # move left
        agent_x -= 1
      elif act_idx == 3:  # move right
        agent_x += 1
      else:  # what? how did you get here?
        raise ValueError("invalid action:", act_idx)

      # "When the agent hits this location, it receives a reward and is immediately transported to a 
      # random location in the maze. Each episode lasts 200 time steps, during which the agent must 
      # accumuluate as much reward as possible. The reward location is fixed within an episode and
      # randomized across episodes."
      #
      # NOTE: this is *very* important. In meta reinforcement learning, an agent is trained to solve
      # a distribution of problems (a set of the same state and action spaces, but with different 
      # reward and transition functions); in the case of the maze navigation task, each of these
      # problems in the same distribution is defined by where the goal is -- hence, a different 
      # reward function. What Miconi et al. have done to make this set of tasks even more challenging
      # is that each time the agent reaches the goal state, it is moved to a new random starting state.
      # Now, there's another layer of learning that needs to be done, i.e., during its lifetime, the
      # agent needs to reinforcement learn to solve a learning problem.

      if self.lab[agent_y, agent_x] == 1:  # punish the agent when it hits the wall
        rew[batch] = MazeNavigationTask.WALL_PEN
      else:
        self.agent_pos[batch] = (agent_y, agent_x)
      if np.array_equal(self.agent_pos[batch], self.goals[batch]):  # Did we hit the reward location? Increase reward and teleport!
        rew[batch] = MazeNavigationTask.GOAL_REW
        pos_y, pos_x = np.random.randint(1, self.lab_size-1, size=2)
        while self.lab[pos_y, pos_x] == 1 or np.array_equal((pos_y, pos_x), self.goals[batch]):
          pos_y, pos_x = np.random.randint(1, self.lab_size-1, size=2)
        self.agent_pos[batch] = (pos_y, pos_x)
    self.prev_rews[:, 0] = rew  # update the previous rewards for the next observation

    # update the next observation
    obs = np.zeros((self.batch_size, self.obs_dim), dtype=np.float32)
    for batch in range(self.batch_size):
      rf = self.lab[self.agent_pos[batch][0]-1:self.agent_pos[batch][0]+2, 
                    self.agent_pos[batch][1]-1:self.agent_pos[batch][1]+2]
      obs[batch] = np.concatenate([rf.ravel(), self.prev_action[batch], self.prev_rews[batch]])
    obs = torch.from_numpy(obs).to(action.device)

    return obs, rew  # no need for `done` and `info`


def main(args):
  if args.debug:
    print("Running in debug mode!")
  else:
    print("Setting up experiment run directory...")
    now_str = ctime().replace(' ', '_')
    exp_run_path = os.path.join(args.exp_path, f'{args.model}_{now_str}')
    if not os.path.isdir(exp_run_path):
      os.makedirs(exp_run_path)
    print(f"{exp_run_path} created!")

    print("Exporting hyperparameter configuration...")
    pprint(args.__dict__)
    conf_path = os.path.join(exp_run_path, 'config.json')
    args_json = json.dumps(args.__dict__, indent=4, sort_keys=True)
    with open(conf_path, 'w') as f:
      f.write(args_json)
    print(f"Experiment configuration saved to {conf_path}")

    progress_filename = os.path.join(exp_run_path, 'progress.csv')
    if args.load_model:
      from shutil import copyfile
      copyfile(os.path.join(args.model_path, args.model, 'progress.csv'), progress_filename)
    else:
      with open(progress_filename, 'w+') as f:
        f.write(f"policy_loss,value_loss,loss,reward\n")

  print("Setting seed for random number generator...")
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  print(f"Building {args.task} environment...")
  if args.task == 'maze':
    env = MazeNavigationTask(args.lab_size, args.batch_size)
  else:
    raise ValueError(f"invalid task type: {args.task}")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Initializing {args.model} agent (device = {device})...")
  if args.model == 'l2rl':  # Learning to Reinforcement Learn
    net = SimpleL2RL(env.obs_dim, env.act_dim, args.hid_dim).to(device)
  elif args.model == 'lstm':  # Learning to Reinforcement Learn (with LSTM)
    net = LstmL2RL(env.obs_dim, env.act_dim, args.hid_dim).to(device)
  elif args.model == 'simple':  # simple neuromodulation
    net = SimpleNeuromodulation(env.obs_dim, env.act_dim, args.hid_dim).to(device)
  elif args.model == 'snlstm':  # simple neuromodulation with LSTM
    net = SimpleNeuromodulationLSTM(env.obs_dim, env.act_dim, args.hid_dim).to(device)
  elif args.model == 'retroactive':  # retroactive neuromodulation with eligibility traces
    net = RetroactiveNeuromodulation(env.obs_dim, env.act_dim, args.hid_dim).to(device)
  else:
    raise ValueError(f"invalid model type: {args.model}")
  
  if args.load_model:  # load a previously saved model
    filename = os.path.join(args.model_path, args.model, 'model.pt')
    print(f"Loading a saved model from {filename}")
    net.load_state_dict(torch.load(filename))
  optimizer = optim.Adam(net.parameters(), lr=args.lr)

  policy_losses = []
  value_losses = []
  losses = []
  total_rewards = []
  global_best = -np.inf

  if args.record and not args.debug:  # don't record stuff in debug mode
    record_path = os.path.join(exp_run_path, 'records')
    if not os.path.isdir(record_path):
      os.makedirs(record_path)
    print(f"Recorded data will be saved to {record_path}")

  print("Starting training!")
  for ep in tqdm(range(args.num_episodes), desc='Episodes'):
    # initialize neuromodulated actor critic internal state
    if args.model == 'l2rl':
      hidden = torch.zeros(args.batch_size, args.hid_dim).to(device)
    elif args.model == 'lstm':
      hidden = (torch.zeros(args.batch_size, args.hid_dim).to(device),
                torch.zeros(args.batch_size, args.hid_dim).to(device))
    elif args.model == 'simple':
      hidden = torch.zeros(args.batch_size, args.hid_dim).to(device)
      hebb = torch.zeros(args.batch_size, *net.W_hh.shape).to(device)
    elif args.model == 'snlstm':
      hidden = (torch.zeros(args.batch_size, args.hid_dim).to(device),
                torch.zeros(args.batch_size, args.hid_dim).to(device))
      hebb = torch.zeros(args.batch_size, *net.cell_W_hh.shape).to(device)
    elif args.model == 'retroactive':  # needs eligibility trace as well
      hidden = torch.zeros(args.batch_size, args.hid_dim).to(device)
      hebb = torch.zeros(args.batch_size, *net.W_hh.shape).to(device)
      trace = torch.zeros(args.batch_size, *net.W_hh.shape).to(device)

    ep_rew = []  # rewards during this episode
    ep_act_prob = []  # action probabilities (pi) in this episode
    ep_act_log_prob = []  # action log probabilities for policy gradient
    ep_entropy = []  # entropies for regularization
    ep_value_pred = []  # value predictions

    if not args.debug and args.record and (ep + 1) % args.rec_freq == 0:
      print("Recording...")  # record initial values
      lab_records = [env.paint_lab()]
      if args.model in ['simple', 'snlstm', 'retroactive']:
        W_records = [net.mod_W_hh(hebb).cpu().detach().numpy()]
        hebb_records = [hebb.cpu().detach().numpy()]
        m_records = [np.zeros((args.batch_size, 1))]

    obs = env.reset(device)

    for i in range(args.max_episode_steps):
      if args.vis and (ep + 1) % args.vis_freq == 0:
        env.render()

      # agent chooses an action
      if args.model == 'l2rl':
        act_prob, value_pred, hidden = net(obs, hidden)
      elif args.model == 'lstm':
        act_prob, value_pred, hidden = net(obs, hidden)
      elif args.model == 'simple':
        act_prob, value_pred, m, hidden, hebb = net(obs, hidden, hebb)
      elif args.model == 'snlstm':
        act_prob, value_pred, m, hidden, hebb = net(obs, hidden, hebb)
      elif args.model == 'retroactive':  # includes eligibility trace to compute the next Hebbian plasticity
        act_prob, value_pred, m, hidden, hebb, trace = net(obs, hidden, hebb, trace)

      dist = distributions.OneHotCategorical(act_prob)
      action = dist.sample()
      act_log_prob = dist.log_prob(action)
      entropy = dist.entropy()

      obs, rew = env.step(action)

      ep_rew.append(rew)
      ep_act_prob.append(act_prob)
      ep_act_log_prob.append(act_log_prob)
      ep_entropy.append(entropy)
      ep_value_pred.append(value_pred)

      if not args.debug and args.record and (ep + 1) % args.rec_freq == 0:
        lab_records.append(env.paint_lab())
        if args.model in ['simple', 'snlstm', 'retroactive']:
          W_records.append(net.mod_W_hh(hebb).cpu().detach().numpy())
          hebb_records.append(hebb.cpu().detach().numpy())
          m_records.append(m.cpu().detach().numpy())

    # Episode is done, now let's do the actual computations of rewards and losses for the A2C algorithm
    policy_loss = 0.0
    value_loss = 0.0
    
    # NOTE: this seems to have improved training efficiency
    GAE = 0.0  # Generalized Advantage Estimation
    R = 0.0  # expected return (utility)
    
    ep_value_pred.append(0.0)
    for t in reversed(range(args.max_episode_steps)):
      rew = torch.from_numpy(ep_rew[t]).unsqueeze(-1).to(device)
      act_log_prob = ep_act_log_prob[t].unsqueeze(-1)
      entropy = ep_entropy[t].unsqueeze(-1)
      
      R = args.gamma * R + rew
      GAE = args.gamma * args.lam * GAE + rew + args.gamma * ep_value_pred[t+1] - ep_value_pred[t]

      policy_loss -= act_log_prob * GAE.detach() - args.entropy_coef * entropy  # maximize
      value_loss += 0.5 * (R - ep_value_pred[t]) ** 2

    policy_loss = torch.mean(policy_loss, dim=0) / args.max_episode_steps
    value_loss = torch.mean(value_loss, dim=0) / args.max_episode_steps
    loss = policy_loss + args.value_coef * value_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
    optimizer.step()

    total_reward = np.mean(sum(ep_rew), axis=0)  # batch-mean of reward sum in this episode
    
    policy_losses.append(policy_loss.item())
    value_losses.append(value_loss.item())
    losses.append(loss.item())
    total_rewards.append(total_reward)

    if (ep + 1) % args.log_freq == 0:
      print(f"================ Episode {ep} ================")
      print("policy loss:", policy_loss.item())
      print("value loss:", value_loss.item())
      print("total loss:", loss.item())
      print("total reward:", total_reward)

    if not args.debug and (ep + 1) % args.save_freq == 0:
      # dump training statistics and refresh their buffer
      with open(progress_filename, 'a') as f:
        for policy_loss_val, value_loss_val, loss_val, reward_val in zip(policy_losses, value_losses, losses, total_rewards):
          f.write(f"{policy_loss_val},{value_loss_val},{loss_val},{reward_val}\n")
      policy_losses = []
      value_losses = []
      losses = []
      total_rewards = []

    if not args.debug and (ep + 1) % args.cp_freq == 0:
      # update the global best score and export the trained model
      if total_reward > global_best:  
        print(f"Saving model due to mean reward increase: {global_best} -> {total_reward}")
        filename = os.path.join(exp_run_path, f'{args.model}.pt')
        torch.save(net.state_dict(), filename)
        print(f"Saved to {filename}!")
        global_best = total_reward

    if not args.debug and args.record and (ep + 1) % args.rec_freq == 0:
      print("Exporting recorded data...")
      ep_record_path = os.path.join(record_path, f'ep_{ep}')
      if not os.path.isdir(ep_record_path):
        os.makedirs(ep_record_path)
      # TODO: save ep_... records as well!
      np.save(os.path.join(ep_record_path, 'lab.npy'), np.stack(lab_records))
      if args.model in ['simple', 'snlstm', 'retroactive']:
        np.save(os.path.join(ep_record_path, 'W.npy'), np.stack(W_records))
        np.save(os.path.join(ep_record_path, 'hebb.npy'), np.stack(hebb_records))
        np.save(os.path.join(ep_record_path, 'm.npy'), np.stack(m_records))


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=0, help='random seed for the experiment')
  parser.add_argument('--debug', action='store_true', default=False, help='enable debug mode')
  parser.add_argument('--task', type=str, default='maze', help='task for the agent to solve')
  parser.add_argument('--model', type=str, default='simple', help='model type to run experiment with')
  parser.add_argument('--load-model', action='store_true', default=False, help='initialize model with a trained model')
  parser.add_argument('--exp-path', type=str, default='./runs', help='path to the directory for experiment results')
  parser.add_argument('--model-path', type=str, default='./models', help='path to the directory for storing trained models')
  parser.add_argument('--hid-dim', type=int, default=100, help='number of hidden neurons in actor critic network')
  parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for actor critic optimizer')
  parser.add_argument('--lab-size', type=int, default=11, help='size of the task environment')
  parser.add_argument('--num-episodes', type=int, default=10000, help='number of training iterations')
  parser.add_argument('--batch-size', type=int, default=16, help='batch size')
  parser.add_argument('--max-episode-steps', type=int, default=200, help='number of time steps in an episode')
  parser.add_argument('--wall-penalty', type=float, default=-0.1, help='penalty for running into a wall')
  parser.add_argument('--goal-reward', type=float, default=10.0, help='reward for reaching the goal')
  parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for return computation')
  parser.add_argument('--lam', type=float, default=1.0, help='lambda parameter for GAE')
  parser.add_argument('--entropy-coef', type=float, default=0.03, help='coefficient for entropy regularization')
  parser.add_argument('--value-coef', type=float, default=0.1, help='coefficient for value prediciton error')
  parser.add_argument('--max-grad-norm', type=float, default=4.0, help='gradient norm clipping')
  parser.add_argument('--vis', action='store_true', default=False, help='visualize the training progress')
  parser.add_argument('--record', action='store_true', default=False, help='record any useful data (as numpy.arrays)')
  parser.add_argument('--log-freq', type=int, default=100, help='log printing frequency (episodes)')
  parser.add_argument('--vis-freq', type=int, default=200, help='visualization frequency (episodes)')
  parser.add_argument('--cp-freq', type=int, default=200, help='model checkpoint freqency (episodes)')
  parser.add_argument('--save-freq', type=int, default=200, help='saving progress frequency (episodes)')
  parser.add_argument('--rec-freq', type=int, default=500, help='recording frequency (episodes)')
  parser.add_argument('--note', type=str, help='any additional note')
  args = parser.parse_args()
  main(args)
