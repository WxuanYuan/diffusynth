import itertools
import json
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tools import create_key
from model.timbre_encoder_pretrain import get_timbre_encoder


class ProjectionLayer(nn.Module):
    """Single-layer Linear projection with dropout, layer norm, and Gelu activation"""

    def __init__(self, input_dim, output_dim, dropout):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ProjectionHead(nn.Module):
    """Stack of 'ProjectionLayer'"""

    def __init__(self, embedding_dim, projection_dim, dropout, num_layers=2):
        super(ProjectionHead, self).__init__()
        self.layers = nn.ModuleList([ProjectionLayer(embedding_dim if i == 0 else projection_dim,
                                                     projection_dim,
                                                     dropout) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class multi_modal_model(nn.Module):
    """The multi-modal model for contrastive learning"""

    def __init__(
            self,
            timbre_encoder,
            text_encoder,
            spectrogram_feature_dim,
            text_feature_dim,
            multi_modal_emb_dim,
            temperature,
            dropout,
            num_projection_layers=1,
            freeze_spectrogram_encoder=True,
            freeze_text_encoder=True,
    ):
        super().__init__()
        self.timbre_encoder = timbre_encoder
        self.text_encoder = text_encoder

        self.multi_modal_emb_dim = multi_modal_emb_dim

        self.text_projection = ProjectionHead(embedding_dim=text_feature_dim,
                                              projection_dim=self.multi_modal_emb_dim, dropout=dropout,
                                              num_layers=num_projection_layers)

        self.spectrogram_projection = ProjectionHead(embedding_dim=spectrogram_feature_dim,
                                                     projection_dim=self.multi_modal_emb_dim, dropout=dropout,
                                                     num_layers=num_projection_layers)

        self.temperature = temperature

        # Make spectrogram_encoder parameters non-trainable
        for param in self.timbre_encoder.parameters():
            param.requires_grad = not freeze_spectrogram_encoder

        # Make text_encoder parameters non-trainable
        for param in self.text_encoder.parameters():
            param.requires_grad = not freeze_text_encoder

    def forward(self, spectrogram_batch, tokenized_text_batch):
        # Getting Image and Text Embeddings (with same dimension)
        spectrogram_features, _, _, _, _ = self.timbre_encoder(spectrogram_batch)
        text_features = self.text_encoder.get_text_features(**tokenized_text_batch)

        # Concat and apply projection
        spectrogram_embeddings = self.spectrogram_projection(spectrogram_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ spectrogram_embeddings.T) / self.temperature
        images_similarity = spectrogram_embeddings @ spectrogram_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        contrastive_loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        contrastive_loss = contrastive_loss.mean()

        return contrastive_loss


    def get_text_features(self, input_ids, attention_mask):
        text_features = self.text_encoder.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_projection(text_features)


    def get_timbre_features(self, spectrogram_batch):
        spectrogram_features, _, _, _, _ = self.timbre_encoder(spectrogram_batch)
        return self.spectrogram_projection(spectrogram_features)


def cross_entropy(preds, targets, reduction='none'):
    """
    Compute the cross-entropy loss between predictions and one-hot targets.

    Args:
        preds (Tensor): Model predictions (logits).
        targets (Tensor): One-hot encoded target labels.
        reduction (str): Specifies the reduction method: 'none' returns loss per sample, 'mean' returns average loss.

    Returns:
        Tensor: The computed cross-entropy loss.
    """
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def get_multi_modal_model(timbre_encoder, text_encoder, model_Config, load_pretrain=False, model_name=None, device="cpu"):
    """
    Initialize and return the multi-modal model using the provided timbre and text encoders.

    Args:
        timbre_encoder (nn.Module): Encoder for timbre/spectrogram.
        text_encoder (nn.Module): Encoder for text.
        model_Config (dict): Configuration parameters for the multi-modal model.
        load_pretrain (bool): Whether to load pretrained weights.
        model_name (str): Name of the model checkpoint.
        device (str): Device to load the model on.

    Returns:
        nn.Module: The initialized multi-modal model.
    """
    mmm = multi_modal_model(timbre_encoder, text_encoder, **model_Config)
    print(f"Model intialized, size: {sum(p.numel() for p in mmm.parameters() if p.requires_grad)}")
    mmm.to(device)

    if load_pretrain:
        print(f"Loading weights from models/{model_name}_MMM.pth")
        checkpoint = torch.load(f'models/{model_name}_MMM.pth', map_location=device)
        mmm.load_state_dict(checkpoint['model_state_dict'])
    mmm.eval()
    return mmm


def train_epoch(text_tokenizer, model, train_loader, labels_mapping, optimizer, device):
    """
    Train the multi-modal model for one epoch on a single batch.

    This function:
      - Retrieves a batch of data and attributes.
      - Ensures that attribute keys are unique.
      - Maps attributes to corresponding text labels.
      - Tokenizes the selected texts.
      - Computes loss and updates model parameters.

    Args:
        text_tokenizer: Tokenizer to convert text to token IDs.
        model (nn.Module): The multi-modal model.
        train_loader (iterable): Training data loader.
        labels_mapping (dict): Mapping from attribute keys to text labels.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        device (str): Device (e.g., "cpu" or "cuda").

    Returns:
        float: Loss value for this batch.
    """
    (data, attributes) = next(iter(train_loader))
    keys = [create_key(attribute) for attribute in attributes]

    # Ensure that each sample has a unique key; if not, get a new batch.
    while(len(set(keys)) != len(keys)):
        (data, attributes) = next(iter(train_loader))
        keys = [create_key(attribute) for attribute in attributes]

    data = data.to(device)

    # Map attributes to their corresponding text labels.
    texts = [labels_mapping[create_key(attribute)] for attribute in attributes]
    # For each sample, randomly select one text from the list.
    selected_texts = [l[random.randint(0, len(l) - 1)] for l in texts]

    # Tokenize the selected texts and move to device.
    tokenized_text = text_tokenizer(selected_texts, padding=True, return_tensors="pt").to(device)

    # Forward pass: compute the loss.
    loss = model(data, tokenized_text)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def valid_epoch(text_tokenizer, model, valid_loader, labels_mapping, device):
    """
    Evaluate the multi-modal model on one validation batch.

    Similar to train_epoch but without backpropagation.

    Args:
        text_tokenizer: Tokenizer to convert text to token IDs.
        model (nn.Module): The multi-modal model.
        valid_loader (iterable): Validation data loader.
        labels_mapping (dict): Mapping from attribute keys to text labels.
        device (str): Device (e.g., "cpu" or "cuda").

    Returns:
        float: Loss value for the validation batch.
    """
    (data, attributes) = next(iter(valid_loader))
    keys = [create_key(attribute) for attribute in attributes]

    # Ensure that attribute keys are unique.
    while(len(set(keys)) != len(keys)):
        (data, attributes) = next(iter(valid_loader))
        keys = [create_key(attribute) for attribute in attributes]

    data = data.to(device)
    texts = [labels_mapping[create_key(attribute)] for attribute in attributes]
    selected_texts = [l[random.randint(0, len(l) - 1)] for l in texts]

    tokenized_text = text_tokenizer(selected_texts, padding=True, return_tensors="pt").to(device)

    loss = model(data, tokenized_text)
    return loss.item()


def train_multi_modal_model(device, training_dataloader, labels_mapping, text_tokenizer, text_encoder,
                            timbre_encoder_Config, MMM_config, MMM_training_config,
                            mmm_name, BATCH_SIZE, max_iter=0, load_pretrain=True,
                            timbre_encoder_name=None, init_loss=None, save_steps=2000):
    """
    Train the multi-modal model (MMM) using the specified configuration and training parameters.

    This function:
      - Initializes the timbre encoder and the multi-modal model.
      - Sets up an optimizer with different learning rates/weight decays for various model parts.
      - Optionally loads pretrained weights.
      - Trains the model for a given number of iterations.
      - Saves checkpoints and hyperparameters periodically.

    Args:
        device (str): Device to run training on.
        training_dataloader (iterable): Data loader for training data.
        labels_mapping (dict): Mapping from attribute keys to text labels.
        text_tokenizer: Tokenizer for processing text labels.
        text_encoder (nn.Module): Text encoder module.
        timbre_encoder_Config (dict): Configuration for the timbre encoder.
        MMM_config (dict): Configuration for the multi-modal model.
        MMM_training_config (dict): Training hyperparameters (e.g., learning rates).
        mmm_name (str): Name used for saving the model.
        BATCH_SIZE (int): Batch size.
        max_iter (int): Maximum training iterations.
        load_pretrain (bool): Whether to load pretrained weights.
        timbre_encoder_name (str, optional): Model name for loading timbre encoder weights.
        init_loss (float, optional): Initial loss value.
        save_steps (int): Interval (in iterations) to save checkpoints.

    Returns:
        tuple: The trained multi-modal model and its optimizer.
    """

    # Helper function to save current hyperparameters and training status.
    def save_model_hyperparameter(model_name, MMM_config, MMM_training_config, BATCH_SIZE, model_size, current_iter,
                                  current_loss):

        model_hyperparameter = MMM_config
        model_hyperparameter.update(MMM_training_config)
        model_hyperparameter["BATCH_SIZE"] = BATCH_SIZE
        model_hyperparameter["model_size"] = model_size
        model_hyperparameter["current_iter"] = current_iter
        model_hyperparameter["current_loss"] = current_loss
        with open(f"models/hyperparameters/{model_name}_MMM.json", "w") as json_file:
            json.dump(model_hyperparameter, json_file, ensure_ascii=False, indent=4)

    # Initialize the timbre encoder (e.g., for spectrogram features).
    timbreEncoder = get_timbre_encoder(timbre_encoder_Config, load_pretrain=True, model_name=timbre_encoder_name,
                                       device=device)

    # Initialize the multi-modal model with the timbre encoder and provided text encoder.
    mmm = multi_modal_model(timbreEncoder, text_encoder, **MMM_config).to(device)

    # Print parameter counts for different parts of the model.
    print(f"spectrogram_encoder parameter: {sum(p.numel() for p in mmm.timbre_encoder.parameters())}")
    print(f"text_encoder parameter: {sum(p.numel() for p in mmm.text_encoder.parameters())}")
    print(f"spectrogram_projection parameter: {sum(p.numel() for p in mmm.spectrogram_projection.parameters())}")
    print(f"text_projection parameter: {sum(p.numel() for p in mmm.text_projection.parameters())}")
    total_parameters = sum(p.numel() for p in mmm.parameters())
    trainable_parameters = sum(p.numel() for p in mmm.parameters() if p.requires_grad)
    print(f"Trainable/Total parameter: {trainable_parameters}/{total_parameters}")

    # Set up optimizer parameters with different learning rates for different model components.
    params = [
        {"params": itertools.chain(
            mmm.spectrogram_projection.parameters(),
            mmm.text_projection.parameters(),
        ), "lr": MMM_training_config["head_lr"], "weight_decay": MMM_training_config["head_weight_decay"]},
    ]
    if not MMM_config["freeze_text_encoder"]:
        params.append({"params": mmm.text_encoder.parameters(), "lr": MMM_training_config["text_encoder_lr"],
                       "weight_decay": MMM_training_config["text_encoder_weight_decay"]})
    if not MMM_config["freeze_spectrogram_encoder"]:
        params.append({"params": mmm.timbre_encoder.parameters(), "lr": MMM_training_config["spectrogram_encoder_lr"],
                       "weight_decay": MMM_training_config["timbre_encoder_weight_decay"]})

    optimizer = torch.optim.AdamW(params, weight_decay=0.)

    # Optionally load pretrained weights for the multi-modal model.
    if load_pretrain:
        print(f"Loading weights from models/{mmm_name}_MMM.pt")
        checkpoint = torch.load(f'models/{mmm_name}_MMM.pth')
        mmm.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("Model initialized.")

    if max_iter == 0:
        print("Return model directly.")
        return mmm, optimizer

    # Evaluate initial validation loss.
    if init_loss is None:
        previous_lowest_loss = valid_epoch(text_tokenizer, mmm, training_dataloader, labels_mapping, device)
    else:
        previous_lowest_loss = init_loss
    print(f"Initial total loss: {previous_lowest_loss}")

    train_loss_list = []
    for i in range(max_iter):

        mmm.train()
        # Train on one batch.
        train_loss = train_epoch(text_tokenizer, mmm, training_dataloader, labels_mapping, optimizer, device)
        train_loss_list.append(train_loss)

        # Get the current training step from optimizer state.
        step = int(
            optimizer.state_dict()['state'][list(optimizer.state_dict()['state'].keys())[0]]['step'].cpu().numpy())
        if (i + 1) % 100 == 0:
            print('%d step' % (step))

        # Periodically save checkpoints.
        if (i + 1) % save_steps == 0:
            current_loss = np.mean(train_loss_list[-save_steps:])
            print(f"train_total_loss: {current_loss}")
            if current_loss < previous_lowest_loss:
                previous_lowest_loss = current_loss
                torch.save({
                    'model_state_dict': mmm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'models/{mmm_name}_MMM.pth')
                save_model_hyperparameter(mmm_name, MMM_config, MMM_training_config, BATCH_SIZE, total_parameters, step,
                                          current_loss)

    return mmm, optimizer