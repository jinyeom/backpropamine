from .miconi_maze import MiconiMaze
from .four_rooms import FourRooms


def make_env(env_id, batch_size, seed=None):
    if env_id == "MiconiMaze":
        return MiconiMaze(batch_size, seed=seed)
    elif env_id == "FourRooms":
        return FourRooms(batch_size, seed=seed)
    else:
        raise NotImplementedError(f"invalid environment name: {env_id}")
