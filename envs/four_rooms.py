import math
import numpy as np
import torch


class FourRooms:
    _LAYOUT = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    _REWARD = 10.0

    def __init__(self, batch_size, seed=None):
        self.batch_size = batch_size
        self.seed(seed=seed)

    @property
    def obs_shape(self):
        return (1, 13, 13)

    @property
    def action_size(self):
        return 4

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed=seed)

    def reset(self):
        self.goal_pos = np.empty((self.batch_size, 2))
        for batch in range(self.batch_size):
            i, j = self.np_random.randint(1, 12, size=2)
            while self._LAYOUT[i, j] == 1 or (i, j) == (3, 3):
                i, j = self.np_random.randint(1, 12, size=2)
            self.goal_pos[batch] = (i, j)

        self.agent_pos = np.array([[3, 3]] * self.batch_size)
        obs = np.array([[self._LAYOUT]] * self.batch_size, dtype=np.float32)
        obs[:, :, 3, 3] = 1.0  # center of the top-left room
        return torch.from_numpy(obs)

    def render(self):
        height, width = self._LAYOUT.shape

        num_rows = int(math.floor(math.sqrt(self.batch_size)))
        num_cols = self.batch_size // num_rows

        mat = []

        for i in range(num_rows):
            row = []

            for j in range(num_cols):
                batch = i * num_cols + j
                canvas = np.zeros((height, width, 3))

                for r in range(height):
                    for c in range(width):
                        if self._LAYOUT[r, c] == 1:
                            canvas[r, c] = (0, 0, 0)
                        elif self._LAYOUT[r, c] == 0:
                            canvas[r, c] = (255, 255, 255)

                goal_pos = tuple(self.goal_pos[batch].astype(np.uint8))
                agent_pos = tuple(self.agent_pos[batch].astype(np.uint8))

                canvas[goal_pos] = (255, 133, 27)  # orange
                canvas[agent_pos] = (0, 116, 217)  # blue

                row.append(canvas)
            mat.append(np.hstack(row))

        rendered = np.vstack(mat)
        scaled = np.kron(rendered, np.ones((8, 8, 1))).astype(np.uint8)

        return scaled

    def step(self, action):
        obs = np.array([[self._LAYOUT]] * self.batch_size, dtype=np.float32)
        reward = np.zeros((self.batch_size, 1), dtype=np.float32)

        for batch in range(self.batch_size):
            i, j = self.agent_pos[batch]
            if action[batch] == 0:
                i -= 1  # up
            elif action[batch] == 1:
                i += 1  # down
            elif action[batch] == 2:
                j -= 1  # left
            elif action[batch] == 3:
                j += 1  # right
            else:
                raise ValueError

            if self._LAYOUT[i, j] == 0:
                self.agent_pos[batch] = (i, j)

            if np.array_equal(self.agent_pos[batch], self.goal_pos[batch]):
                reward[batch] = self._REWARD

                # teleport the agent to a random position
                i, j = self.np_random.randint(1, 12, size=2)
                while self._LAYOUT[i, j] == 1:
                    i, j = self.np_random.randint(1, 12, size=2)
                self.agent_pos[batch] = (i, j)

            # update the observation
            i, j = self.agent_pos[batch]
            obs[batch, :, i, j] = 1.0

        obs = torch.from_numpy(obs)
        reward = torch.from_numpy(reward)

        return obs, reward
