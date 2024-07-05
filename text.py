import numpy as np
from dataclasses import dataclass
import librosa
import matplotlib.pyplot as plt
import torch
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2Model

import wave
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from tqdm import trange
from sklearn import preprocessing

# faceQs modules
from video import Media
from dataset import DatasetFS
from dataset_utils import *
from utils import *


class Text(Media):
    """
    Handles audio streams in a Dataset.
    """
    def __init__(self, config, dataset: DatasetFS, filename, store, store_info=True, verbose=0):
        super().__init__(config, dataset, filename, store, store_info, verbose)
        

    def merge_info_dataset(self) -> None:
        if self.store_info:
            verbatimT(self.verbose, 1, "Text info extraction... ")
            info_dict_all = []
            for index in range(self.dataset.size()):
                info_dict_all.append(self.get_text_info(index))
            verbatimT(self.verbose, 1, "done.\n")
        else:
            verbatimT(self.verbose, 1, "Text info not stored.\n")

        # merge Series into DataFrame
        self.data_frame["info"] = pd.Series(info_dict_all)

    def get_text_info(self, index: int = None) -> dict:
        videopath = self.dataset.get_filename(index, full=True)
        return {"videopath": videopath}

    def set_dataset(self, dataset: DatasetFS):
        """
        Sets a dataset.
        """
        self.dataset = dataset

    def load_text(self, index: int) -> None:

        textpath = self.dataset.get_filename(index, full=True)
        # audiopath = "/var/data/student_home/agnelli/new_dataset/output/RAVDESS/demo.txt"
        with open(textpath, "r") as f:
            self.text = f.read()
