import torch
import torch.nn as nn

class Contrastive_model(nn.Module):
    """
    The Contrastive_model class is a PyTorch module that implements a contrastive learning model for video, audio, and text inputs. The model consists of three embedding modules, one for each input modality, that project the inputs into a shared embedding space. The model then computes the contrastive loss between the embeddings of the different modalities using the CLIP loss function.
    
    The forward method of the model takes in video, audio, and text inputs (the text input is optional) and returns the embeddings for each modality, as well as the overall contrastive loss. The embeddings are L2-normalized before being used to compute the contrastive loss.
    
    The model also includes a learnable logit scale parameter that is used to scale the logits before computing the contrastive loss.
    """
    def __init__(self, hidden_channels, out_channels):
        super(Contrastive_model, self).__init__()

        self.embedding_video = nn.Sequential(nn.Linear(400, hidden_channels), 
                                             nn.ReLU(), 
                                             nn.Linear(hidden_channels, hidden_channels), 
                                             nn.ReLU(), 
                                             nn.Linear(hidden_channels, out_channels))
        self.embedding_audio = nn.Sequential(nn.Linear(768, hidden_channels), 
                                             nn.ReLU(), 
                                             nn.Linear(hidden_channels, hidden_channels),
                                             nn.ReLU(), 
                                             nn.Linear(hidden_channels, out_channels))
        self.embedding_text = nn.Sequential(nn.Linear(768, hidden_channels), 
                                            nn.ReLU(), 
                                            nn.Linear(hidden_channels, hidden_channels), 
                                            nn.ReLU(), 
                                            nn.Linear(hidden_channels, out_channels))
        logit_scale_init_value=2.6592
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))


    def forward(self, x_video, x_audio, x_text = None):
        x_video = self.embedding_video(x_video)
        if x_audio is not None:
            x_audio = self.embedding_audio(x_audio)
        if x_text is not None:
            x_text = self.embedding_text(x_text)

        x_video = nn.functional.normalize(x_video)
        if x_audio is not None:
            x_audio= nn.functional.normalize(x_audio)
        if x_text is not None:
            x_text= nn.functional.normalize(x_text) 

        logit_scale = self.logit_scale.exp()
        if x_audio is not None:
            logits_audio_text = torch.matmul(x_audio, x_audio.t()) * logit_scale
        if x_text is not None:  
            logits_video_text = torch.matmul(x_video, x_text.t()) * logit_scale
        if x_audio is not None:
            logits_audio_video = torch.matmul(x_video, x_audio.t()) * logit_scale

        loss = 0

        if x_audio is not None and x_text is not None:
            loss = clip_loss(logits_audio_text)
        if x_text is not None:
            loss += clip_loss(logits_video_text)
        if x_audio is not None:
            loss += clip_loss(logits_audio_video)

        return x_video, x_audio, x_text, loss
    
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the contrastive loss for a given set of logits.
    
    The contrastive loss is calculated as the cross-entropy loss between the logits and a target tensor containing the sequence of indices from 0 to the length of the logits tensor.
    
    Args:
        logits (torch.Tensor): A 2D tensor containing the logits for the contrastive task.
    
    Returns:
        torch.Tensor: The contrastive loss.
    """
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    Computes the contrastive loss for a given similarity matrix.
    
    The contrastive loss is calculated as the average of the cross-entropy loss for the rows (caption loss) and the columns (image loss) of the similarity matrix.
    
    Args:
        similarity (torch.Tensor): A 2D tensor containing the similarity scores between captions and images.
    
    Returns:
        torch.Tensor: The contrastive loss.
    """
        
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

    """
    Downstream model that uses a pre-trained contrastive model to extract features from video and audio inputs, and then applies a classifier to produce the final output.
    
    The model first loads the pre-trained contrastive model from a specified path, and then applies a classifier on the concatenated video and audio features to produce the final output.
    
    Args:
        hidden_channels (int): The number of hidden channels in the contrastive model.
        out_channels (int): The number of output channels in the classifier.
    
    Returns:
        torch.Tensor: The final output of the downstream model.
    """
    def __init__(self, hidden_channels):
        super(Downstream_model_regressive, self).__init__()

        self.classifier = nn.Sequential(nn.Linear(1168, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, 1))
        # self.classifier = nn.Sequential(nn.Linear(1168, out_channels))

    def forward(self, x_video, x_audio):

        x = self.classifier(torch.cat((x_audio, x_video), dim=1))

        return x