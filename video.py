import numpy as np
import cv2
import pandas as pd
from tqdm import trange


import torch
from torchvision import transforms

# faceQs modules
from dataset import DatasetFS
from media import Media
from utils import *
from dataset_utils import *

class Video(Media):
    """
    Handles video streams in a Dataset.
    """
    def __init__(self, config, dataset: DatasetFS, filename, store, store_info=True, verbose=0):
        super().__init__(config, dataset, filename, store, store_info, verbose)

    def merge_info_dataset(self):

        if self.store_info:
            info_dict_all = []
            verbatimT(1, 1, "Merging video info...")
            for index in trange(self.dataset.size(), desc="   "):
                info_dict_all.append(self.get_video_info(index))
            verbatimT(1, 1, "done.\n")
        else:
            verbatimT(self.verbose, 1, "Video info not stored.\n")

        # merge Series into DataFrame
        self.data_frame["info"] = pd.Series(info_dict_all)

    def get_video_info(self, index=None) -> dict:
        videopath = self.dataset.get_filename(index, full=True)
        vidCap = cv2.VideoCapture(videopath)
        height = vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vidCap.get(cv2.CAP_PROP_FRAME_WIDTH)
        totalFrames = int(vidCap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidCap.get(cv2.CAP_PROP_FPS)
        duration = totalFrames / fps
        return {"videopath": videopath, "height": height, "width": width, "totalFrames": totalFrames, "fps": fps, "duration": duration}

    def get_frames(self, index=None, videopath=None):
        """
        Yields the frames of a video.
        """
        if index:
            videopath = self.dataset.get_filename(index, full=True)
        if videopath:
            vidcap = cv2.VideoCapture(videopath)
            while True:
                success, frame = vidcap.read()
                if not success:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # set RGB colorspace for the frame
                yield frame.astype(np.uint8)
            vidcap.release()

        def __iter__(self):
            self.n = 0
            return self

        def __next__(self):
            if self.n < self.dataset.size():
                self.n += 1
                video = self.dataset.get_filename(self.n, full=True)
                return self.extract_frames(videopath=video)
            else:
                raise StopIteration

    def load_video_frames(self, videopath=None, index=None, max_frames=3000):
        
        frames = []
        count = 0

        if index is not None:
            videopath = self.dataset.get_filename(index, full=True)
        if videopath:
            vidcap = cv2.VideoCapture(videopath)
        
        while vidcap.isOpened() and count < max_frames:
            ret, frame = vidcap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #cv2.COLOR_BGR2GRAY
            frames.append(frame)
            count += 1

        vidcap.release()
        self.frames = frames

    def preprocess_frames(self, frame_size=(224, 224)):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize(frame_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5], std=[0.5]),])

        processed_frames = [transform(frame) for frame in self.frames]
        return torch.stack(processed_frames)[::6,:,:,:]  # Shape: (num_frames, 3, height, width)
