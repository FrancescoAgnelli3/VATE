import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import *
#from swin_transformer import SwinTransformer3D
#from torchvision.models.video import mvit_v1_b
from contrastive_model import Contrastive_model
from transformers import VivitImageProcessor, BertTokenizer
from VATE import VATEDataset
from tqdm import trange
from video import Video
from audio import Audio
from text import Text
from torch.utils.data import Dataset
import pickle 
import os
from torch.utils.data import DataLoader
from transformers import VivitForVideoClassification, BertModel
import torchaudio
# from torch_geometric.data import Data
from train_test import train_test_contrastive
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):
        self.TASK_NAME = "Emotion"
        self.ROOT = os.path.dirname(os.path.abspath(__file__))
        self.DATASET = VATEDataset
        self.DATASET_NAME = "VATE"
        self.OUTPUT_DIR = f"{self.ROOT}/output/{self.DATASET_NAME}"

        self.DATASET_ARGS = {
            "dataset_name": self.DATASET_NAME,
            "data_path": f"{self.ROOT}/feature_extraction/VATE",
            "store": True,
            "shuffle": True,
            "pkl_fname": f"{self.DATASET_NAME}_data_frame.pkl",
        }
        self.MEDIA = {
            "Audio": {"class": Audio, "ext": "wav", "store_info": False, "store": True, "pkl_fname": f"{self.DATASET_NAME}_audio_media_frame.pkl"},
            "Text": {"class": Text, "ext": "txt", "store_info": False, "store": True, "pkl_fname": f"{self.DATASET_NAME}_text_media_frame.pkl"},    
            "Video": {"class": Video, "ext": "mp4", "store_info": False, "store": True, "pkl_fname": f"{self.DATASET_NAME}_video_media_frame.pkl"},    
}
        
class LoaderDataset(Dataset):
    def __init__(self, x_fold_video, x_fold_audio, x_fold_text):
        self.x_fold_video = x_fold_video
        self.x_fold_audio = x_fold_audio
        self.x_fold_text = x_fold_text

    def __len__(self):
        return len(self.x_fold_video)

    def __getitem__(self, idx):
        item_video = self.x_fold_video[idx]
        item_audio = self.x_fold_audio[idx]
        item_text = self.x_fold_text[idx]
        return item_video, item_audio, item_text

class Training_contrastive():
    def __init__(self, config, store):
        self.config = config
        self.store = store

    def training(self):
        num_epochs = 200
        batch_size = 1024
        learning_rate = 0.01

        image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = Contrastive_model(200, 100)

        model_video = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
        bundle = torchaudio.pipelines.HUBERT_BASE
        model_audio = bundle.get_model()
        model_text = BertModel.from_pretrained("bert-base-uncased")

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, verbose=True)

        VATE = VATEDataset(self.config, ext = 'mp4', verbose=1)
        video_media = Video(
                    self.config, dataset=VATE, filename=config.MEDIA["Video"]["pkl_fname"], store=config.MEDIA["Video"]["store"], store_info=config.MEDIA["Video"]["store_info"], verbose=1
                )
        VATE = VATEDataset(config, ext = 'wav', verbose=1)
        audio_media = Audio(
                    self.config, dataset=VATE, filename=config.MEDIA["Audio"]["pkl_fname"], store=config.MEDIA["Audio"]["store"], store_info=config.MEDIA["Audio"]["store_info"], verbose=1
                )
        VATE = VATEDataset(config, ext = 'txt', verbose=1)
        text_media = Text(
                    self.config, dataset=VATE, filename=config.MEDIA["Text"]["pkl_fname"], store=config.MEDIA["Text"]["store"], store_info=config.MEDIA["Text"]["store_info"], verbose=1
                )

        x_train_loader = VATE.train_test_split()

        if self.store:
            train_loader_video = []
            train_loader_audio = []
            train_loader_text = []
            for i in trange(len(x_train_loader)):
                    video_media.load_video_frames(index = x_train_loader[i])
                    item = np.array(video_media.frames)
                    item = frame_resampling_np(item, 32)
                    item = image_processor(list(item), return_tensors="pt")
                    with torch.no_grad():
                        item["pixel_values"] = item["pixel_values"].squeeze(1)
                        item = model_video(**item).logits.squeeze(0)
                    train_loader_video.append(item)

                    audio_media.load_audio(index = x_train_loader[i])
                    item_audio = audio_media.compute_feature_hubert()
                    with torch.no_grad():
                        item_audio, _ = model_audio.extract_features(item_audio)
                        item_audio = item_audio[-1][0].mean(0)
                    train_loader_audio.append(item_audio)

                    text_media.load_text(index = x_train_loader[i])
                    item_txt = torch.tensor([text_tokenizer.encode(text_media.text)])
                    with torch.no_grad():
                        item_txt = model_text(item_txt).pooler_output.squeeze(0)
                    train_loader_text.append(item_txt)


            train_loader = LoaderDataset(train_loader_video, train_loader_audio, train_loader_text)
            pathout = os.path.join(config.OUTPUT_DIR, "train_loader.pkl")
            print("Storing train loader...")
            with open(pathout, 'wb') as handle:
                pickle.dump(train_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Train loader stored into file: " + pathout)
        else:
            print("Loading train loader...")
            pathout = os.path.join(config.OUTPUT_DIR, "train_loader.pkl")
            with open(pathout, 'rb') as handle:
                train_loader = pickle.load(handle)

        train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)

        Training = train_test_contrastive(optimizer, device, scheduler, self.config)
        Training.training(num_epochs, train_loader, model)

if __name__ == "__main__":
    config = Config()
    store = False
    tr_contrastive = Training_contrastive(config, store)
    tr_contrastive.training()
    



