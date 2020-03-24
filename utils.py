import json
import logging
import math
from pathlib import Path
import sys
import uuid

import torch
from torch.utils.tensorboard import SummaryWriter
import pyglet
from pyglet import gl
from pyglet.window import Window
from tabulate import tabulate
from termcolor import colored


class Experiment:
    def __init__(self, args):
        self.args = args

        self.artifact_path = Path(args.artifact_path or "/tmp")
        self.exp_id = f"{args.env_id}-{uuid.uuid4().hex}"

        self.exp_path = self.artifact_path / self.exp_id
        self.exp_path.mkdir(parents=True, exist_ok=False)

        # set up logger
        self.logger = logging.getLogger(self.exp_id)
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(self.exp_path / "progress.log")
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]:\n%(message)s\n")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # set up TensorBoard
        self.writer = SummaryWriter(self.exp_path)

        # current best performance
        self.curr_best_score = -math.inf

        colored_exp_id = colored(self.exp_id, "green")
        print(f"Starting experiment: {colored_exp_id}")

    def show_args(self):
        table = [(k, v) for k, v in self.args.__dict__.items()]
        print(tabulate(table, tablefmt="fancy_grid", stralign="center", numalign="center"))

    def export_args(self):
        with open(self.exp_path / "config.json", "w") as f:
            f.write(json.dumps(self.args.__dict__, indent=4, sort_keys=True))

    def log(self, global_step, **kwargs):
        keys, values = tuple(zip(*kwargs.items()))
        keys = ["global_step"] + list(keys)
        values = [global_step] + list(values)

        self.logger.info(
            tabulate(
                [values],
                headers=keys,
                tablefmt="fancy_grid",
                floatfmt=".5f",
                stralign="center",
                numalign="center",
            )
        )

        for k, v in kwargs.items():
            self.writer.add_scalar(k, v, global_step=global_step)

    def checkpoint(self, global_step, model, optimizer, loss):
        checkpoint = {
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        checkpoint_path = self.exp_path / f"checkpoint_{global_step}.pth"
        torch.save(checkpoint, checkpoint_path)

    def update_best(self, model, score):
        if score > self.curr_best_score:
            self.logger.info(f"Improvement detected: {self.curr_best_score:.5f} --> {score:.5f}")
            final_model_path = self.exp_path / f"model_final.pth"
            torch.save(model.state_dict(), final_model_path)
            self.curr_best_score = score


class NumpyTube:
    def __init__(self):
        self.window = None
        self.isopen = False

    def __del__(self):
        self.close()

    def imshow(self, img, caption=None):
        height, width, _ = img.shape
        pitch = -3 * width

        if self.window is None:
            self.window = Window(width=width, height=height, vsync=False)
            self.width = width
            self.height = height
            self.isopen = True

        data = img.tobytes()
        image = pyglet.image.ImageData(width, height, "RGB", data, pitch=pitch)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        texture = image.get_texture()
        texture.width = self.width
        texture.height = self.height

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        texture.blit(0, 0)
        self.window.flip()

        if caption is not None:
            self.window.set_caption(caption)

    def close(self):
        if self.isopen:
            self.window.close()
            self.window = None
            self.isopen = False
