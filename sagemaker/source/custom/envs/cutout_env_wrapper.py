import numpy as np
from ray.tune import registry

try:
    from envs.procgen_env_wrapper import ProcgenEnvWrapper
except ModuleNotFoundError:
    from custom.envs.procgen_env_wrapper import ProcgenEnvWrapper

class CutoutEnvWrapper(ProcgenEnvWrapper):
    """
    Adopted from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    """
    def __init__(self, config, num_patches=5, patch_len=16):
        super().__init__(config)
        self.num_patches = num_patches
        self.patch_len = patch_len

    def observation(self, observation):
        h, w = observation.shape[:2]
        half_len = self.patch_len // 2
        for _ in range(self.num_patches):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - half_len, 0, h)
            y2 = np.clip(y + half_len, 0, h)
            x1 = np.clip(x - half_len, 0, w)
            x2 = np.clip(x + half_len, 0, w)
            r = np.random.randint(256)
            g = np.random.randint(256)
            b = np.random.randint(256)
            color = np.array([r, g, b], dtype=uint8)
            observation[y1:y2, x1:x2] = color
        return observation
