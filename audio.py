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


class Audio(Media):
    """
    Handles audio streams in a Dataset.
    """
    def __init__(self, config, dataset: DatasetFS, filename, store, store_info=True, verbose=0):
        super().__init__(config, dataset, filename, store, store_info, verbose)
        
        self.set_MFCC_params(hop_length=33.34, win_size=1024)

    def merge_info_dataset(self) -> None:
        if self.store_info:
            verbatimT(self.verbose, 1, "Audio info extraction... ")
            info_dict_all = []
            for index in range(self.dataset.size()):
                info_dict_all.append(self.get_audio_info(index))
            verbatimT(self.verbose, 1, "done.\n")
        else:
            verbatimT(self.verbose, 1, "Audio info not stored.\n")

        # merge Series into DataFrame
        self.data_frame["info"] = pd.Series(info_dict_all)

    def get_audio_info(self, index: int = None) -> dict:
        videopath = self.dataset.get_filename(index, full=True)
        obj = wave.open(videopath, "r")
        num_channels = obj.getnchannels()
        samp_width = obj.getsampwidth()
        sample_rate = obj.getframerate()
        num_frames = obj.getnframes()
        obj.close()
        return {"videopath": videopath, "num_channels": num_channels, "samp_width": samp_width, "sample_rate": sample_rate, "num_frames": num_frames}

    def set_dataset(self, dataset: DatasetFS):
        """
        Sets a dataset.
        """
        self.dataset = dataset

    def set_MFCC_params(self, hop_length: float = 33.3, win_size=2048, n_mfcc=13, delta1=True, delta2=True):
        """
        Sets the MFCC params. Librosa uses centered frames,
        so that the kth frame is centered around sample k * hop_length.

        Args:
           - hop_length (float): time in ms (milliseconds)
           - win_size (int): audio_length of the FFT window.
           - n_mfcc (int): number of MFCCs to return.
           - delta1 (bool): whether to compute also delta1 or not
           - delta2 (bool): whether to compute aldo deltae or not
        """
        self.hop_length = hop_length
        self.win_size = win_size
        self.n_mfcc = n_mfcc
        self.delta1 = delta1
        self.delta2 = delta2

    def load_audio(self, index: int) -> None:
        """
        Extracts the MFCC of the specified audio file.

        Args:
          index (int): index of the audio file within the dataset.
        """
        audiopath = self.dataset.get_filename(index, full=True)
        # self.audio_data, self.sample_rate = librosa.load(audiopath)
        self.audio_data, self.sample_rate =  torchaudio.load(audiopath)
        self.num_channels = self.audio_data.ndim

    def compute_feature_hubert(self):

        bundle = torchaudio.pipelines.HUBERT_BASE
        waveform = torchaudio.functional.resample(self.audio_data, self.sample_rate, bundle.sample_rate)

        return waveform

    def compute_feature_wav2vec(self):
        
        model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        processor = Wav2Vec2Processor.from_pretrained(model_id,)
        model = Wav2Vec2Model.from_pretrained(model_id).to("cuda")
        target_sampling_rate = processor.feature_extractor.sampling_rate
        self.feature = processor(self.audio_data, sampling_rate=target_sampling_rate, return_tensors="pt", padding=True)
        input_values = self.feature.input_values.to("cuda")
        attention_mask = self.feature.attention_mask.to("cuda")

        with torch.no_grad():
            self.feature = model(input_values, attention_mask=attention_mask)
        self.feature = self.feature["extract_features"].squeeze(0).cpu()
        
    def compute_raw(self) -> None:
        """
        Extracts the MFCC of the specified audio file. Librosa uses centered frames,
        so that the kth frame is centered around sample k * hop_length.
        """

        # compute MFCC features from the raw signal
        hop_length_sample = int(self.hop_length * self.sample_rate / 1000.0)
        x = librosa.feature.melspectrogram(y=self.audio_data, sr=self.sample_rate, hop_length= hop_length_sample)
        x = x.transpose(1, 0)
        # t, c = x.shape
        # window = 10
        # x = np.row_stack([np.zeros((window - 2, c)),x])
        # x = np.row_stack([x,np.zeros((window - 2, c))])
        # x2 = np.zeros((t, window, c))
        # for i in range(t):
        #     x2[i] = x[i:i+window]
        
        # and the first and second-order differences (delta features)
        self.feature = np.float32(x)
    
    def length_audio(self) -> None:
        """
        Extracts the MFCC of the specified audio file. Librosa uses centered frames,
        so that the kth frame is centered around sample k * hop_length.
        """

        # compute MFCC features from the raw signal
        return librosa.get_duration(y=self.audio_data, sr=self.sample_rate)

    def compute_mfcc(self) -> None:
        """
        Extracts the MFCC of the specified audio file. Librosa uses centered frames,
        so that the kth frame is centered around sample k * hop_length.
        """

        # compute MFCC features from the raw signal
        hop_length_sample = int(self.hop_length * self.sample_rate / 1000.0)
        mfcc = librosa.feature.mfcc(y=self.audio_data, sr=self.sample_rate, hop_length=hop_length_sample, n_fft=self.win_size, n_mfcc=self.n_mfcc)
        # and the first and second-order differences (delta features)
        self.mfcc = mfcc
        if self.delta1:
            mfcc_delta1 = librosa.feature.delta(mfcc, order=1)
            self.mfcc = np.vstack((self.mfcc, mfcc_delta1))
            self.mfcc_delta1 = mfcc_delta1
        if self.delta2:
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            self.mfcc = np.vstack((self.mfcc, mfcc_delta2))
            self.mfcc_delta2 = mfcc_delta2
        self.mfcc = preprocessing.StandardScaler().fit_transform(self.mfcc)  # (x-mu)/sigma
        # self.mfcc = np.absolute(self.mfcc)
        # max_mfcc = self.mfcc.max()
        # min_mfcc = self.mfcc.min()
        # self.mfcc = (2 * self.mfcc - max_mfcc - min_mfcc) / (max_mfcc - min_mfcc)  # minmax normalization on abs value
        self.mfcc = self.mfcc.transpose(1, 0)
        self.feature = np.float32(self.mfcc)

    def extract_all_stream(self, use = "wav2vec") -> None:
        """
        Extract MFCC for a subset of (all) audio.
        """
        MFCC_all = []
        numer = 0
        # loop on all videos
        verbatimT(1, 1, "Audio MFCC extraction...")
        for idx in trange(self.dataset.size(), desc="   "):
            self.load_audio(idx)
            if use == "vgg":
                self.compute_vgg(idx)
            elif use == "mfcc":
                self.compute_mfcc()
            elif use == "wav2vec":
                self.compute_feature_wav2vec()
            elif use == "raw":
                self.compute_raw()
            elif use == "length_audio":
                numer = max(self.length_audio(), numer)
            MFCC_all.append(self.feature)
        verbatimT(1, 1, "-------\n")
        return MFCC_all

    def get_frame_audio(self, index, win_size, hop_size, centered=True):
        """
        Extracts ovelapped frames at multiple hops and of fixed audio_length.

        Args:
           - index (int): index of the audio file within the dataset.
           - win_size (int): window size in frames
           - hop_size (float): hops in frames
           - centered (logical): if true windows is centered around hops
        """
        if not hasattr(self, "audio_data"):
            self.load_audio(index)
        if self.num_channels == 1:
            audio_length = self.audio_data.shape[0]
        else:
            audio_length = self.audio_data.shape[1]
        if centered:
            frame_num = np.floor((audio_length - 1) / hop_size).astype(int)
            self.frames = np.zeros((frame_num, win_size))
            # set frames
            for k in range(frame_num):
                left = int(max(int((k + 1) * hop_size - win_size / 2), 0))
                right = int(min(int((k + 1) * hop_size + win_size / 2), audio_length))
                self.frames[k, 0 : right - left] = self.audio_data[left:right]
        else:
            frame_num = np.floor(audio_length / hop_size + 1).astype(int)
            self.frames = np.zeros((frame_num, win_size))
            # set frames
            for k in range(frame_num):
                left = k * hop_size
                right = min(left + win_size, audio_length)
                if right - left < win_size:
                    self.frames[k, 0 : right - left] = self.audio_data[left:right]
                else:
                    self.frames[k] = self.audio_data[left:right]

    def show_mfcc(self, scale: float = 1) -> None:
        """
        Creates an interactive imshow of a list of frames. Must be called
        inside a Jupyter notebook or a Colab notebook.

        Args:
          index (int): index of video.
         scale (float): a rescale factor (<=1).
        """
        fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
        img1 = librosa.display.specshow(self.mfcc, x_axis="time", ax=ax[0])
        ax[0].set(title="MFCC")
        fig.colorbar(img1, ax=[ax[0]])
        img2 = librosa.display.specshow(self.mfcc_delta1, x_axis="time", ax=ax[1])
        ax[1].set(title="MFCC (delta1)")
        fig.colorbar(img2, ax=[ax[1]])
        img3 = librosa.display.specshow(self.mfcc_delta2, x_axis="time", ax=ax[1])
        ax[2].set(title="MFCC (delta2)")
        fig.colorbar(img3, ax=[ax[2]])

    def interactive_frame_plot(self, index, win_size, hop_size):
        """
        Creates an interactive imshow of a list of audio frames. Must be called
        inside a Jupyter notebook or a Colab notebook.

        Args:
           index (int): index of video.
           win_size (int): window size in frames
           hop_size (float): hops in frames
        """
        self.get_frame_audio(index, win_size, hop_size)
        if self.verbose:
            verbatimT(text=f"Frame shape (num frame, frame size) = ({self.frames.shape})")
        n_frame = self.frames.shape[0]

        # definition of the plot "callback function"
        def plot_f(x):
            plt.figure(figsize=(8, 3))
            plt.plot(self.frames[x], linewidth=2)
            plt.title("audio frame")
            plt.xlabel("samples")

        # generate our user interface.
        interact(plot_f, x=IntSlider(min=0, max=n_frame - 1, value=1))
