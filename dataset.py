import os
from typing import Iterable
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import random
import numpy as np

# faceQs modules
from utils import *
from dataset_utils import *


class DatasetFS(ABC):
    """
    Represents datasets of three types: 'Video', Audio', 'Image', handling the
    filesystem image and using a Pandas data frame to collect files paths.
    The following series are automatically computed for each dataset:
       - filenames = file name of each dataset item
       - fullpath  = full path of each dataset item
       - classes   = class name (default) of each dataset item (shoud be replaced)
       - key       = a unique label of each dataset item
       - media     = 'Video', Audio' or 'Image'
    """

    def __init__(self, args, ext: str = None, verbose=1) -> None:
        self.args = args
        self.verbose = verbose
        self.ext = ext

        # scans filesystem
        if self.args.DATASET_ARGS["store"]:
            print(f"Reading Dataset from FS {self.args.DATASET_ARGS['data_path']} and saving into {self.args.OUTPUT_DIR}/{self.args.DATASET_ARGS['pkl_fname']}")
            self.scan()
            self.store_dataset()
        else:
            if not os.path.exists(self.args.DATASET_ARGS["data_path"]):
                raise FileNotFoundError(f"Dataset dataframe file {self.args.DATASET_ARGS['data_path']} not found. Please run again with -s flag to compute it.")
            self.load_dataset()

        # print info
        if self.verbose:
            self.info()

    def info(self) -> None:
        """
        Dataset description.
        """
        verbatimT(1, 1, f"Dataset:    {self.args.DATASET_NAME}")
        verbatimT(1, 1, f"media type: {[key for key in self.args.MEDIA.keys()]}", deep=1)
        verbatimT(1, 1, f"num items:  {self.size()}:", deep=1)
        verbatimT(1, 1, f"series:     {list(self.data_frame.keys())}:", deep=1)
        verbatimT(1, 1, "-------\n")

    def scan(self) -> None:
        """
        Scans the filsystem pointed by datapath
        """
        verbatimT(self.verbose, 1, "Scan dataset ... ")
        filenames = []
        fullpath = []
        classes = []
        media = []
        key = []
        for dirpath, _, files in os.walk(self.args.DATASET_ARGS["data_path"], topdown=False):
            verbatimT(self.verbose, 1, f"Found directory: {dirpath}", deep=1)
            ext = self.ext
            for file_name in files:
                if file_name.split(".")[-1] == ext:
                    ext = file_name.split(".")[-1]
                    fullpath.append(dirpath)
                    key.append(os.path.join(dirpath, file_name.replace("." + ext, "")))
                    filenames.append(file_name)
                    tail = os.path.split(dirpath)[-1]
                    classes.append(tail)
                    media.append([key for key in self.args.MEDIA.keys() if self.args.MEDIA[key]["ext"] == ext][0])
                    verbatimT(self.verbose, 2, f"class: {tail}, fullname: {file_name}, fullpath: {dirpath}", deep=2)

        # put on pandas (sorted by fullpath)
        data_descriptor = {"key": key, "class": classes, "fullpath": fullpath, "filename": filenames, "media": media}
        self.data_frame = pd.DataFrame(data_descriptor)
        self.data_frame.sort_values(by=["key"], inplace=True)
        self.data_frame = self.data_frame.reset_index(drop=True)
        verbatimT(self.verbose, 2, "done.\n")

    def shuffle(self, seed: int = 0) -> None:
        # shuffle the DataFrame rows
        self.data_frame = self.data_frame.sample(frac=1, random_state=seed)

    def data_selection(self, subset: list[int]) -> None:
        """
        Dataset selection (reduction) based on a index list.

        Args:
           - subset: list of indexes to select
        """
        self.data_frame = self.data_frame.iloc[subset]
        self.data_frame = self.data_frame.reset_index(drop=True)

    def select_by_range(self, interval: tuple[int, int] = None) -> None:
        """
        Selects dataset itmes (filenames) by range.

        Args:
           - interval: 2-len tuple used as range.
        """
        # select the subset
        if len(interval) >= 2:
            subset = range(interval[0], interval[1])
        else:
            warn(f"The length of {interval} must be => 2!")
        # subset
        self.data_selection(subset)

    def select_by_fullname(self, subset: list = None) -> None:
        # TODO
        pass

    def filter(self, filter: str = None, filter_value: Iterable | str = None):
        """
        Filter the dataset by a given filter and filter_value.
        """
        if filter is None or filter_value is None:
            raise ValueError("Please specify a filter and a filter_value")

        if filter not in self.data_frame.columns:
            raise ValueError(f"Filter {filter} not in dataframe columns")

        if isinstance(filter_value, str):
            filter_value = [filter_value]

        self.data_frame = self.data_frame[self.data_frame[filter].isin(filter_value)]

        print(f"Dataset filtered by {filter} = {filter_value}, new size: {self.size()}")

    def load_dataset(self, filename: Path = None) -> None:
        """
        Load dataset descriptor form file.

        Args:
           - filename: full or relative path to store dataset frame
        """
        if filename:
            pathout = filename
        else:
            pathout = self.args.OUTPUT_DIR
            pathout = set_path(pathout, self.args.DATASET_ARGS["pkl_fname"], task="join")
        # check
        if not set_path(pathout, task="check"):
            # warn(f"File {pathout} not found!")
            raise FileNotFoundError(f"File {pathout} not found!")
        try:
            with open(pathout) as f:
                self.data_frame = pd.read_pickle(pathout)
        except Exception as e:
            raise e
            # warn(f"Impossible to load Pandas file {pathout}!")

        verbatimT(self.verbose, 1, "Dataset descriptor loaded from file: " + pathout)
        verbatimO(self.verbose, 2, self.data_frame)

    def store_dataset(self):
        """
        Stores dataset descriptor to file

        Args:
           - filename: full or relative path to store dataset frame
        """

        pathout = os.path.join(self.args.ROOT, self.args.OUTPUT_DIR, self.args.DATASET_ARGS["pkl_fname"])
        self.data_frame.to_pickle(pathout)
        verbatimT(self.verbose, 1, "Dataset descriptor stored into file: " + pathout)

    def data_iter(self, serie: str) -> None:
        """
        Return a map of the specified data serie.
        """
        return self.data_frame[serie].__iter__()

    def get_data_serie(self, serie: str) -> list:
        """
        Return a specified dataset serie as list.

        Args:
           - serie: the specified Pandas serie
        """
        return self.data_frame[serie].to_list()

    def get_filename(self, index: int, full: bool = True) -> str:
        """
        Return the (complete) file name with specified index.
        """
        if index < self.size():
            filepath = self.data_frame.iloc[index, 3]
            if full:
                fullpath = self.data_frame.iloc[index, 2]
                filepath = set_path(fullpath, filepath, task="join")
            return filepath
        else:
            warn(f"Error! Index: {index} exceeds the max: {self.dataset.size()}")

    def size(self) -> int:
        """
        Return data size.
        """
        return self.data_frame.shape[0]

    def __iter__(self):
        self.n = -1
        return self

    def __next__(self):
        if self.n < self.size():
            self.n += 1
            return self.data_frame.iloc[self.n]
        else:
            raise StopIteration

    @abstractmethod
    def set_classes(self) -> None:
        """
        Set spefic class names.
        """
        pass

    @abstractmethod
    def train_test_split(self, train_perc: int = 80) -> list[list]:
        """
        Builds the training and test sets.

        Args:
           - train_perc: percentage in [0, 100] for the training set, the complement is for test set
        """
        pass


class RAVDESSDataset(DatasetFS):
    """
    Class describing the dataset RAVDESS.

    Filename example: 02-01-06-01-02-01-12.mp4
                      M -V- E -E -S- R -A
    - Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    - Vocal channel (01 = speech, 02 = song).
    - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    - Emotional intensity (01 = normal, 02 = strong). There is no strong intensity for the 'neutral' emotion.
    - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    - Repetition (01 = 1st repetition, 02 = 2nd repetition).
    - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
    """

    def __init__(self, args: Args, media_type="Video", scan=True, load=False, save=False, verbose=0):
        # super class's constructor
        super().__init__(args, media_type, scan, load, save, verbose)

        # set class names
        if scan:
            self.set_classes()

    # @abstractmethod
    def set_classes(self):
        emotions = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
        classes = []
        actors = []
        ext = self.media_args["ext"]
        for fname in self.data_frame["filename"]:
            fname = fname.replace("." + ext, "")
            s = fname.split("-")
            classes.append(emotions[s[2]])
            actors.append(s[6])

        # replaces classes in the serie
        self.data_frame["class"] = classes
        self.classes = classes
        # add actors
        self.data_frame["actor"] = actors
        self.actors = actors

    def train_test_split(self, train_fold: list = None, train_perc: int = None, leave_actor: int = None, five_fold=0) -> list[list]:
        """ "
        Builds the training and test sets.

        Args:
           - train_perc: percentage in [0, 100] for the training set, the complement is for test set
           - leave_actor: number of the actor to be left out of the training test
           - five_fold: number of the fold to be left out of the training set

        Return:
           - X_train_list: training list of actors
           - X_test_list: test list of actors
           - y_train_list: training class labels
           - y_test_list: test class labels
        """
        Fold_0 = ["02", "05", "14", "15", "16"]
        Fold_1 = ["03", "06", "07", "13", "18"]
        Fold_2 = ["10", "11", " 12", "19", "20"]
        Fold_3 = ["08", "17", "21", "23", "24"]
        Fold_4 = ["01", "04", "09", "22"]
        Fold = [Fold_0, Fold_1, Fold_2, Fold_3, Fold_4]

        try:
            actors = self.get_data_serie("actor")
        except:
            warn(f"Exec method set_classes() before!")

        # collect data in train, test lists
        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []

        # use percentage to split actors
        if train_fold is None and train_perc is not None:
            if int(train_perc) > 1 or int(train_perc) < 0:
                warn(f"The vale for {train_perc} must be in [0,1]")
            else:
                actor_uniq = np.sort(list(set(actors)))
                random.shuffle(actor_uniq)
                train_fold = actor_uniq[0 : int(train_perc * len(actor_uniq))]
        else:
            if leave_actor:
                actor_uniq = np.sort(list(set(actors)))
                random.shuffle(actor_uniq)
                mask = actor_uniq != leave_actor
                train_fold = np.sort(actor_uniq[mask])
            else:
                if five_fold > -1:
                    actor_uniq = np.sort(list(set(actors)))
                    random.shuffle(actor_uniq)
                    train_fold = np.sort([item for item in actor_uniq if item not in Fold[five_fold]])
                else:
                    warn(f"A splitting choice must be made")
        # use a given fold to split actors
        if train_fold is not None:
            for i in range(self.size()):
                # print(actors[i],train_fold )
                if actors[i] in train_fold:
                    X_train_list.append(i)
                    y_train_list.append(self.data_frame.iloc[i, 1])
                else:
                    X_test_list.append(i)
                    y_test_list.append(self.data_frame.iloc[i, 1])

        # final shuffle
        permute = np.random.permutation(len(X_train_list)).tolist()
        X_train_list = np.array(X_train_list)[permute].tolist()
        y_train_list = np.array(y_train_list)[permute].tolist()

        permute = np.random.permutation(len(X_test_list)).tolist()
        X_test_list = np.array(X_test_list)[permute].tolist()
        y_test_list = np.array(y_test_list)[permute].tolist()

        # return
        return X_train_list, X_test_list, y_train_list, y_test_list
