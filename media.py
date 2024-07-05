import pandas as pd

from abc import ABC, abstractmethod

# faceQs modules
from dataset import DatasetFS
from dataset_utils import *
from utils import *


class Media(ABC):
    def __init__(self, config, dataset: DatasetFS, filename, store, store_info=True, verbose=0):
        self.args = config
        self.filename = filename
        self.verbose = verbose
        self.store_info = store_info

        self.dataset = dataset
        self.data_frame = dataset.data_frame  # main data container (Pandas)
        if store_info:
            self.merge_info_dataset()

        if store:
            self.store_media_data()
        else:
            self.load_media_data()

    def info(self):
        """
        Dataset description.
        """
        verbatimT(1, 1, f"Dataset: {self.args.DATASET_NAME}")
        verbatimT(1, 1, f"size: {self.dataset.size()}:", deep=1)
        verbatimT(1, 1, f"series: {list(self.data_frame.keys())}:", deep=1)
        verbatimT(1, 1, "done.\n")

    def set_dataset(
        self,
        dataset: DatasetFS,
        store_info=True,
    ) -> None:
        """
        Sets a new dataset to process for the Media object.
        """
        self.dataset = dataset
        self.store_info = store_info
        self.data_frame = dataset.data_frame  # main container (Pandas)
        if store_info:
            self.merge_info_dataset()

    def load_media_data(self) -> None:
        """
        Load dataset descriptor form file

        Args:
           - store_path: full or relative path to store dataset frame
           - media_type: the media type
           - filename: a file to load data from
        """
        assert self.filename is not None, "Filename must be provided"
        assert self.filename.endswith(".pkl"), "Filename must be a .pkl file"

        pathout = os.path.join(self.args.ROOT, self.args.OUTPUT_DIR, self.filename)
        if set_path(pathout, task="check"):
            self.data_frame = pd.read_pickle(pathout)
            verbatimT(self.verbose, 1, "Dataset descriptor loaded from file: " + pathout)
            verbatimO(self.verbose, 1, self.data_frame)

    def store_media_data(self):
        """
        Stores dataset descriptor to file
        - data (self.data_frame) is stored as Pandas

        Args:
           - store_path: full or relative path to store dataset frame
           - media_type: the media type
           - filename: a file in which to store data
        """
        pathout = os.path.join(self.args.ROOT, self.args.OUTPUT_DIR, self.filename)
        self.data_frame.to_pickle(pathout)
        verbatimT(self.verbose, 1, "Media descriptor stored into file: " + pathout)

    @abstractmethod
    def merge_info_dataset():
        """
        Merges all Media info in the dataset Dataset as a dict.
        The main container to merge is self.data_frame
        """
        pass
