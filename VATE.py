import os
from typing import Iterable
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import random
import numpy as np

from collections import defaultdict
from dataset import DatasetFS
from utils import *

class VATEDataset(DatasetFS):
    """
    Class describing the dataset VATE.

    Filename example: Question_01.mp4
    """

    def __init__(self, args: Args, ext: str, verbose=0):
        # super class's constructor
        super().__init__(args, ext, verbose)

        self.info_path = os.path.join(args.DATASET_ARGS["data_path"], "info.txt")

        # set class names
        if args.DATASET_ARGS["store"]:
            # self.shuffle()
            self.store_dataset()
            if self.args.DATASET_ARGS["shuffle"]:
                print("The dataset has been shuffled")
                self.shuffle()
        else:
            if self.args.DATASET_ARGS["shuffle"]:
                print("The dataset has been shuffled")
                self.shuffle()

    # @abstractmethod
    def set_classes(self):
        emotions = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
        classes = []
        actors = []
        for fname in self.data_frame["filename"]:
            fname = fname.split(".")[-2]
            s = fname.split("-")
            classes.append(emotions[s[2]])
            actors.append(s[6])

        # replaces classes in the serie
        self.data_frame["class"] = classes
        self.classes = classes
        # add actors
        self.data_frame["actor"] = actors
        self.actors = actors

    def train_test_split(self) -> list[list]:
        """
        Splits the dataset into training and test sets.

        Returns:
            list[list]: A list containing the indices of the training samples.
        """

        # collect data in train lists
        X_train_list = []

        for i in range(self.size()):
            X_train_list.append(i)

        # final shuffle
        permute = np.random.permutation(len(X_train_list)).tolist()
        X_train_list = np.array(X_train_list)[permute].tolist()

        # return
        return X_train_list
