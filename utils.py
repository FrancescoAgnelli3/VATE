import warnings
from dataclasses import dataclass
import sys
from enum import Enum
import torch
import numpy as np


@dataclass
class Args:  # used as abstract type for sharing args
    pass


@dataclass
class TemplateArgs:
    """
    DataClass used to store general project parameters for the dataset.
    """

    facesQs_path: str = "path-to-source"  # root to faceQs source (for devel)
    rootdir: str = ""  # root dir of the project
    data_path: str = ""  # path to data
    device: str = "cuda"  # gpu accelerator if available
    data_fname: str = ""  # filename of the data description

    def add_root_path(self):  # add root dir to sys.path
        sys.path.append(self.rootdir)


def verbatimT(verbose, true, text, deep=0):

    if verbose >= 1 and true == 1:
        if deep == 0:
            pre = "-> "
        if deep == 1:
            pre = "   "
        print(pre + text)
    elif verbose >= 2 and true == 2:
        pre = "   "
        print(pre + text + "\n")


def verbatimO(verbose, obj, level=0):
    if verbose == 2:
        print(obj)


def warn(text):
    warnings.warn(text)


class Task(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1


class Split(Enum):
    PERCENTAGE_SPLIT = 0
    CROSS_VALIDATION = 1
    RAVDESS_SPLIT = 2
    RAVDESS_SPLIT_5F = 3


class TrainEvaluationMetrics(Enum):
    ACCURACY = 0
    LOSS = 1

def frame_resampling(x, max_frame = 60, method="pad"):
    if len(x) > max_frame:
        return x[0:max_frame]  # cut
    elif method == "pad":
        diff = max_frame - len(x)
        x = torch.vstack((x, torch.zeros(diff, x.shape[1], x.shape[2], x.shape[3])))
        return x
    
def frame_resampling_np(x, max_frame = 60, method="pad"):
    if len(x) > max_frame:
        return x[0:max_frame]  # cut
    elif method == "pad":
        diff = max_frame - len(x)
        x = np.vstack((x, np.zeros((diff, x.shape[1], x.shape[2], x.shape[3]))))
        return x
