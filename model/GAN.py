import json
import numpy as np
import torch
from torch import nn
from six.moves import xrange
from torch.utils.tensorboard import SummaryWriter
import random

from model.diffusion import ConditionedUnet
from tools import create_key

class Discriminator(nn.Module):
    def __init__(self, label_emb_dim):
        """
        Initialize the Discriminator network.

        Args:
            label_emb_dim (int): Dimension of the label embedding.
        """

        super(Discriminator, self).__init__()
        # Convolutional layers to extract features from the input image.
        # Input has 4 channels; these layers progressively downsample the feature map.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Adaptive pooling to generate fixed-size features
            nn.Flatten()
        )

        # Text embedding module to process label embeddings.
        self.text_embedding = nn.Sequential(
            nn.Linear(label_emb_dim, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final fully connected layer to output a single scalar
        # The input dimension is 512 (from image features) + 512 (from text embedding)
        self.fc = nn.Linear(512 + 512, 1)

    def forward(self, x, text_emb):
        """
        Forward pass of the Discriminator.

        Args:
            x (Tensor): Input image tensor.
            text_emb (Tensor): Text label embedding.

        Returns:
            Tensor: The discriminator output score.
        """
        # Extract features from the image using convolutional layers.
        x = self.conv_layers(x)
        # Process text embedding.
        text_emb = self.text_embedding(text_emb)
        # Concatenate image and text features along the channel dimension.
        combined = torch.cat((x, text_emb), dim=1)
        output = self.fc(combined)
        return output



def evaluate_GAN(device, generator, discriminator, iterator, encodes2embeddings_mapping):
    """
    Evaluate the performance of the GAN by calculating the accuracy of the discriminator
    on both real and fake images.

    Args:
        device (str): Device to run the evaluation on.
        generator (nn.Module): Generator network.
        discriminator (nn.Module): Discriminator network.
        iterator (iterable): Data iterator yielding (data, attributes).
        encodes2embeddings_mapping (dict): Mapping from attribute keys to embedding vectors.

    Returns:
        tuple: Average accuracy for real images and fake images.
    """
    # Move models to the specified device and set them to evaluation mode.
    generator.to(device)
    discriminator.to(device)
    generator.eval()
    discriminator.eval()

    real_accs = []
    fake_accs = []

    # Disable gradient calculation for evaluation.
    with torch.no_grad():
        for i in range(100):
            # Get a batch of data and its associated attributes.
            data, attributes = next(iter(iterator))
            data = data.to(device)

            # For each attribute, obtain the corresponding embedding and randomly select one embedding per sample.
            conditions = [encodes2embeddings_mapping[create_key(attribute)] for attribute in attributes]
            selected_conditions = [random.choice(conditions_of_one_sample) for conditions_of_one_sample in conditions]
            selected_conditions = torch.stack(selected_conditions).float().to(device)

            # Real images and their corresponding label embeddings.
            real_images = data.to(device)
            labels = selected_conditions.to(device)

            # Generate fake images using random noise as input.
            noise = torch.randn_like(real_images).to(device)
            fake_images = generator(noise)

            # Get discriminator predictions for real and fake images.
            real_preds = discriminator(real_images, labels).reshape(-1)
            fake_preds = discriminator(fake_images, labels).reshape(-1)
            # Calculate accuracy: for real images, prediction should be > 0.5; for fake images, < 0.5.
            real_acc = (real_preds > 0.5).float().mean().item()
            fake_acc = (fake_preds < 0.5).float().mean().item()

            real_accs.append(real_acc)
            fake_accs.append(fake_acc)

    # Compute average accuracy over all iterations.
    average_real_acc = sum(real_accs) / len(real_accs)
    average_fake_acc = sum(fake_accs) / len(fake_accs)

    return average_real_acc, average_fake_acc


def get_Generator(model_Config, load_pretrain=False, model_name=None, device="cpu"):
    """
    Initialize the generator (Conditioned U-Net) and optionally load pretrained weights.

    Args:
        model_Config (dict): Configuration for the U-Net generator.
        load_pretrain (bool): Whether to load pretrained weights.
        model_name (str): Model name used for loading weights.
        device (str): Device to load the model on.

    Returns:
        nn.Module: The generator network.
    """
    generator = ConditionedUnet(**model_Config)
    print(f"Model intialized, size: {sum(p.numel() for p in generator.parameters() if p.requires_grad)}")
    generator.to(device)

    if load_pretrain:
        print(f"Loading weights from models/{model_name}_generator.pth")
        checkpoint = torch.load(f'models/{model_name}_generator.pth', map_location=device)
        generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()
    return generator


def get_Discriminator(model_Config, load_pretrain=False, model_name=None, device="cpu"):
    """
    Initialize the discriminator and optionally load pretrained weights.

    Args:
        model_Config (dict): Configuration for the discriminator.
        load_pretrain (bool): Whether to load pretrained weights.
        model_name (str): Model name used for loading weights.
        device (str): Device to load the model on.

    Returns:
        nn.Module: The discriminator network.
    """
    discriminator = Discriminator(**model_Config)
    print(f"Model intialized, size: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad)}")
    discriminator.to(device)

    if load_pretrain:
        print(f"Loading weights from models/{model_name}_discriminator.pth")
        checkpoint = torch.load(f'models/{model_name}_discriminator.pth', map_location=device)
        discriminator.load_state_dict(checkpoint['model_state_dict'])
    discriminator.eval()
    return discriminator


def train_GAN(device, init_model_name, unetConfig, BATCH_SIZE, lr_G, lr_D, max_iter, iterator, load_pretrain,
                     encodes2embeddings_mapping, save_steps, unconditional_condition, uncondition_rate, save_model_name=None):
    """
    Train the GAN model with both generator and discriminator.

    Args:
        device (str): Device to perform training.
        init_model_name (str): Initial model name used for checkpointing.
        unetConfig (dict): Configuration for the generator (Conditioned U-Net).
        BATCH_SIZE (int): Batch size.
        lr_G (float): Learning rate for the generator.
        lr_D (float): Learning rate for the discriminator.
        max_iter (int): Maximum training iterations.
        iterator (iterable): Data iterator yielding (data, attributes).
        load_pretrain (bool): Whether to load pretrained weights.
        encodes2embeddings_mapping (dict): Mapping from attributes to embeddings.
        save_steps (int): Interval (in steps) to save checkpoints.
        unconditional_condition (Tensor/array): Unconditional condition used with a given probability.
        uncondition_rate (float): Probability of using the unconditional condition.
        save_model_name (str, optional): Name for saving the model; defaults to init_model_name if not provided.

    Returns:
        tuple: Generator, Discriminator, and their respective optimizers.
    """
    if save_model_name is None:
        save_model_name = init_model_name

    # Helper function to save model hyperparameters to a JSON file.
    def save_model_hyperparameter(model_name, unetConfig, BATCH_SIZE, model_size, current_iter, current_loss):
        model_hyperparameter = unetConfig
        model_hyperparameter["BATCH_SIZE"] = BATCH_SIZE
        model_hyperparameter["lr_G"] = lr_G
        model_hyperparameter["lr_D"] = lr_D
        model_hyperparameter["model_size"] = model_size
        model_hyperparameter["current_iter"] = current_iter
        model_hyperparameter["current_loss"] = current_loss
        with open(f"models/hyperparameters/{model_name}_GAN.json", "w") as json_file:
            json.dump(model_hyperparameter, json_file, ensure_ascii=False, indent=4)

    generator = ConditionedUnet(**unetConfig)
    discriminator = Discriminator(unetConfig["label_emb_dim"])
    generator_size = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    discriminator_size = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)

    print(f"Generator trainable parameters: {generator_size}, discriminator trainable parameters: {discriminator_size}")
    generator.to(device)
    discriminator.to(device)
    optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=lr_G, amsgrad=False)
    optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr_D, amsgrad=False)

    # Optionally load pretrained weights for both generator and discriminator.
    if load_pretrain:
        print(f"Loading weights from models/{init_model_name}_generator.pt")
        checkpoint = torch.load(f'models/{init_model_name}_generator.pth')
        generator.load_state_dict(checkpoint['model_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loading weights from models/{init_model_name}_discriminator.pt")
        checkpoint = torch.load(f'models/{init_model_name}_discriminator.pth')
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("Model initialized.")
    # If no training iterations are set, return the models immediately.
    if max_iter == 0:
        print("Return model directly.")
        return generator, discriminator, optimizer_G, optimizer_D

    train_loss_G, train_loss_D = [], []
    writer = SummaryWriter(f'runs/{save_model_name}_GAN')

    # Define loss criterion: Binary Cross Entropy with Logits.
    criterion = nn.BCEWithLogitsLoss()
    generator.train()

    # Main training loop.
    for i in xrange(max_iter):
        # Fetch a batch of data and associated attributes.
        data, attributes = next(iter(iterator))
        data = data.to(device)

        # For each attribute, obtain its corresponding embedding.
        conditions = [encodes2embeddings_mapping[create_key(attribute)] for attribute in attributes]
        unconditional_condition_copy = torch.tensor(unconditional_condition, dtype=torch.float32).to(device).detach()
        selected_conditions = [unconditional_condition_copy if random.random() < uncondition_rate else random.choice(
            conditions_of_one_sample) for conditions_of_one_sample in conditions]
        batch_size = len(selected_conditions)
        selected_conditions = torch.stack(selected_conditions).float().to(device)

        # Assign real images and labels.
        real_images = data.to(device)
        labels = selected_conditions.to(device)

        # Create real and fake labels (ones for real, zeros for fake).
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ================== Train Discriminator ==================
        optimizer_D.zero_grad()

        # Discriminator loss on real images.
        outputs_real = discriminator(real_images, labels)
        loss_D_real = criterion(outputs_real, real_labels)

        # Generate fake images using noise as input to the generator.
        noise = torch.randn_like(real_images).to(device)
        fake_images = generator(noise, labels)

        # Discriminator loss on fake images (detach to avoid gradients flowing to generator).
        outputs_fake = discriminator(fake_images.detach(), labels)
        loss_D_fake = criterion(outputs_fake, fake_labels)

        # Total discriminator loss.
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()

        # ================== Train Generator ==================
        optimizer_G.zero_grad()

        # Generator loss: try to fool the discriminator so that fake images are classified as real.
        outputs_fake = discriminator(fake_images, labels)
        loss_G = criterion(outputs_fake, real_labels)
        loss_G.backward()
        optimizer_G.step()

        # Record training losses.
        train_loss_G.append(loss_G.item())
        train_loss_D.append(loss_D.item())
        # Retrieve the current training step from the optimizer state.
        step = int(optimizer_G.state_dict()['state'][list(optimizer_G.state_dict()['state'].keys())[0]]['step'].numpy())

        if (i + 1) % 100 == 0:
            print('%d step' % (step))

        # Periodically save checkpoints and log losses.
        if (i + 1) % save_steps == 0:
            current_loss_D = np.mean(train_loss_D[-save_steps:])
            current_loss_G = np.mean(train_loss_G[-save_steps:])
            print('current_loss_G: %.5f' % current_loss_G)
            print('current_loss_D: %.5f' % current_loss_D)

            writer.add_scalar(f"current_loss_G", current_loss_G, step)
            writer.add_scalar(f"current_loss_D", current_loss_D, step)

            # Save generator checkpoint.
            torch.save({
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer_G.state_dict(),
            }, f'models/{save_model_name}_generator.pth')
            save_model_hyperparameter(save_model_name, unetConfig, BATCH_SIZE, generator_size, step, current_loss_G)
            torch.save({
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer_D.state_dict(),
            }, f'models/{save_model_name}_discriminator.pth')
            save_model_hyperparameter(save_model_name, unetConfig, BATCH_SIZE, discriminator_size, step, current_loss_D)

        # Additionally, save historical checkpoints every 10,000 steps.
        if step % 10000 == 0:
            torch.save({
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer_G.state_dict(),
            }, f'models/history/{save_model_name}_{step}_generator.pth')
            torch.save({
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer_D.state_dict(),
            }, f'models/history/{save_model_name}_{step}_discriminator.pth')

    return generator, discriminator, optimizer_G, optimizer_D


