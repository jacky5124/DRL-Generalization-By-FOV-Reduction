import cv2 as cv
import numpy as np
from ray.tune import registry

try:
    from envs.procgen_env_wrapper import ProcgenEnvWrapper
except ModuleNotFoundError:
    from custom.envs.procgen_env_wrapper import ProcgenEnvWrapper

class ZoomInEnvWrapper(ProcgenEnvWrapper):
    def __init__(self, config, factor=1.5):
        super().__init__(config)
        self.factor = factor
        if self.factor < 1.0:
            raise Exception("zoom_in factor is less than 1, making it zoom out!")

    def observation(self, observation):
        old_len = observation.shape[0]
        new_len = int(old_len / self.factor)
        off_len = (old_len - new_len) // 2
        boundary = off_len + new_len
        zoomed_in = observation[off_len:boundary, off_len:boundary]
        zoomed_in = cv.resize(zoomed_in, observation.shape[:2])
        return zoomed_in
