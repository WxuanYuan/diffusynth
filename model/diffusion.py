import json
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from six.moves import xrange
from torch.utils.tensorboard import SummaryWriter
import random

from metrics.IS import get_inception_score
from tools import create_key

from model.diffusion_components import default, ConvNextBlock, ResnetBlock, SinusoidalPositionEmbeddings, Residual, \
    PreNorm, \
    Downsample, Upsample, exists, q_sample, get_beta_schedule, pad_and_concat, ConditionalEmbedding, \
    LinearCrossAttention, LinearCrossAttentionAdd


class ConditionedUnet(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim=None,
            down_dims=None,
            up_dims=None,
            mid_depth=3,
            with_time_emb=True,
            time_dim=None,
            resnet_block_groups=8,
            use_convnext=True,
            convnext_mult=2,
            attn_type="linear_cat",
            n_label_class=11,
            condition_type="instrument_family",
            label_emb_dim=128,
    ):
        """
        A conditional U-Net architecture for diffusion models.

        This U-Net integrates conditional embeddings and time embeddings, and supports
        both ResNet and ConvNeXt style blocks, as well as different types of attention.

        Parameters:
            in_dim (int): Number of input channels.
            out_dim (int, optional): Number of output channels. Defaults to in_dim if not provided.
            down_dims (list, optional): List of channel dimensions for the downsampling path.
            up_dims (list, optional): List of channel dimensions for the upsampling path.
            mid_depth (int): Depth of the bottleneck part.
            with_time_emb (bool): Whether to use time embeddings.
            time_dim (int, optional): Dimension of the time embedding. Defaults to 4 * down_dims[0].
            resnet_block_groups (int): Number of groups for ResNet blocks.
            use_convnext (bool): Whether to use ConvNeXt blocks instead of ResNet blocks.
            convnext_mult (int): Multiplicative factor for ConvNeXt blocks.
            attn_type (str): Type of attention to use ("linear_cat" or "linear_add").
            n_label_class (int): Number of label classes for conditioning.
            condition_type (str): Type of condition (e.g., "instrument_family").
            label_emb_dim (int): Dimension of the label embedding.
        """
        super().__init__()

        # Create a conditional embedding layer for labels (with an extra token, hence n_label_class+1)
        self.label_embedding = ConditionalEmbedding(int(n_label_class + 1), int(label_emb_dim), condition_type)

        # Set default channel dimensions if not provided
        if up_dims is None:
            up_dims = [128, 128, 64, 32]
        if down_dims is None:
            down_dims = [32, 32, 64, 128]

        out_dim = default(out_dim, in_dim)
        # Ensure the dimensions of downsampling and upsampling paths match appropriately
        assert len(down_dims) == len(up_dims), "len(down_dims) != len(up_dims)"
        assert down_dims[0] == up_dims[-1], "down_dims[0] != up_dims[-1]"
        assert up_dims[0] == down_dims[-1], "up_dims[0] != down_dims[-1]"
        down_in_out = list(zip(down_dims[:-1], down_dims[1:]))
        up_in_out = list(zip(up_dims[:-1], up_dims[1:]))
        time_dim = default(time_dim, int(down_dims[0] * 4))

        # Initial convolution: project input to the first downsampling dimension
        self.init_conv = nn.Conv2d(in_dim, down_dims[0], 7, padding=3)

        # Select the block type based on the flag "use_convnext"
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # Choose the attention module based on the attn_type parameter
        if attn_type == "linear_cat":
            attn_klass = partial(LinearCrossAttention)
        elif attn_type == "linear_add":
            attn_klass = partial(LinearCrossAttentionAdd)
        else:
            raise NotImplementedError()

        # Time embedding MLP: projects sinusoidal time embeddings to a higher dimensional space
        if with_time_emb:
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(down_dims[0]),
                nn.Linear(down_dims[0], time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # Build the downsampling (encoder) path
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        skip_dims = []  # Used to store dimensions for skip connections

        # For each downsampling block, build two convolutional blocks (with attention) and a downsample layer
        for down_dim_in, down_dim_out in down_in_out:
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(down_dim_in, down_dim_out, time_emb_dim=time_dim),

                        Residual(PreNorm(down_dim_out, attn_klass(down_dim_out, label_emb_dim=label_emb_dim, ))),
                        block_klass(down_dim_out, down_dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(down_dim_out, attn_klass(down_dim_out, label_emb_dim=label_emb_dim, ))),
                        Downsample(down_dim_out),
                    ]
                )
            )
            skip_dims.append(down_dim_out)

        # Bottleneck layers between encoder and decoder
        mid_dim = down_dims[-1]
        self.mid_left = nn.ModuleList([])
        self.mid_right = nn.ModuleList([])

        # Build multiple blocks for the bottleneck
        for _ in range(mid_depth - 1):
            self.mid_left.append(block_klass(mid_dim, mid_dim, time_emb_dim=time_dim))
            self.mid_right.append(block_klass(mid_dim * 2, mid_dim, time_emb_dim=time_dim))
        self.mid_mid = nn.ModuleList(
            [
                block_klass(mid_dim, mid_dim, time_emb_dim=time_dim),
                Residual(PreNorm(mid_dim, attn_klass(mid_dim, label_emb_dim=label_emb_dim, ))),
                block_klass(mid_dim, mid_dim, time_emb_dim=time_dim),
            ]
        )

        # Build the upsampling (decoder) path
        for ind, (up_dim_in, up_dim_out) in enumerate(up_in_out):
            # Retrieve corresponding skip connection dimension (from downsampling)
            skip_dim = skip_dims.pop()
            self.ups.append(
                nn.ModuleList(
                    [
                        # Concatenate skip connection with current feature map and process
                        # pop&cat (h/2, w/2, down_dim_out)
                        block_klass(up_dim_in + skip_dim, up_dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(up_dim_in, attn_klass(up_dim_in, label_emb_dim=label_emb_dim, ))),
                        Upsample(up_dim_in),
                        # Process after concatenation
                        # pop&cat (h, w, down_dim_out)
                        block_klass(up_dim_in + skip_dim, up_dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(up_dim_out, attn_klass(up_dim_out, label_emb_dim=label_emb_dim, ))),
                        # Final block in this upsampling stage
                        # pop&cat (h, w, down_dim_out)
                        block_klass(up_dim_out + skip_dim, up_dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(up_dim_out, attn_klass(up_dim_out, label_emb_dim=label_emb_dim, ))),
                    ]
                )
            )

        # Final convolution: merge last skip connection with decoded features and project to output channels
        self.final_conv = nn.Sequential(
            block_klass(down_dims[0] + up_dims[-1], up_dims[-1]), nn.Conv2d(up_dims[-1], out_dim, 3, padding=1)
        )

    def size(self):
        """
        Print the total number of parameters and the number of trainable parameters.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")


    def forward(self, x, time, condition=None):
        """
        Forward pass of the Conditioned U-Net.

        Parameters:
            x (tensor): Input image tensor.
            time (tensor): Time steps for the diffusion process.
            condition (tensor, optional): Conditioning labels.

        Returns:
            x (tensor): Output tensor after U-Net processing.
        """
        # Obtain condition embedding if a condition is provided
        if condition is not None:
            condition_emb = self.label_embedding(condition)
        else:
            condition_emb = None

        h = []  # List to store features for skip connections

        # Initial convolution on the input image
        x = self.init_conv(x)
        h.append(x)

        # Compute time embeddings if available
        time_emb = self.time_mlp(time) if exists(self.time_mlp) else None

        # Downsampling (encoder) path with intermediate skip connections
        for block1, attn1, block2, attn2, downsample in self.downs:
            x = block1(x, time_emb)
            x = attn1(x, condition_emb)
            h.append(x)
            x = block2(x, time_emb)
            x = attn2(x, condition_emb)
            h.append(x)
            x = downsample(x)
            h.append(x)

        # Bottleneck processing
        for block in self.mid_left:
            x = block(x, time_emb)
            h.append(x)

        (block1, attn, block2) = self.mid_mid
        x = block1(x, time_emb)
        x = attn(x, condition_emb)
        x = block2(x, time_emb)

        for block in self.mid_right:
            # Merge with corresponding skip connection using padding and concatenation
            x = pad_and_concat(h.pop(), x)
            x = block(x, time_emb)

        # Upsampling (decoder) path with skip connections
        for block1, attn1, upsample, block2, attn2, block3, attn3 in self.ups:
            x = pad_and_concat(h.pop(), x)
            x = block1(x, time_emb)
            x = attn1(x, condition_emb)
            x = upsample(x)

            x = pad_and_concat(h.pop(), x)
            x = block2(x, time_emb)
            x = attn2(x, condition_emb)

            x = pad_and_concat(h.pop(), x)
            x = block3(x, time_emb)
            x = attn3(x, condition_emb)

        # Final merge with the last skip connection and final convolution
        x = pad_and_concat(h.pop(), x)
        x = self.final_conv(x)
        return x


def conditional_p_losses(denoise_model, x_start, t, condition, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                         noise=None, loss_type="l1"):
    """
    Compute the conditional loss for the diffusion model.

    This function first adds noise to the original image (x_start) using the q_sample function,
    then predicts the noise using the denoising model and computes the loss between the true noise
    and the predicted noise.

    Parameters:
        denoise_model: The neural network model predicting noise.
        x_start (tensor): Original input image.
        t (tensor): Diffusion time steps.
        condition (tensor): Conditioning information.
        sqrt_alphas_cumprod (array): Precomputed square roots of cumulative alpha values.
        sqrt_one_minus_alphas_cumprod (array): Precomputed square roots of (1 - cumulative alpha values).
        noise (tensor, optional): Optional noise tensor; if None, random noise is generated.
        loss_type (str): Loss function type ("l1", "l2", or "huber").

    Returns:
        loss (tensor): Computed loss value.
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    # Generate a noisy version of x_start according to the diffusion process
    x_noisy = q_sample(x_start=x_start, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                       sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
    # Predict the noise using the denoising model
    predicted_noise = denoise_model(x_noisy, t, condition)

    # Compute loss based on the specified type
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


def evaluate_diffusion_model(device, model, iterator, BATCH_SIZE, timesteps, unetConfig, encodes2embeddings_mapping,
                             uncondition_rate, unconditional_condition):
    """
    Evaluate the diffusion model by computing the average loss over 500 iterations.

    For each iteration, a batch is retrieved from the iterator, conditioning embeddings are selected
    (with a chance to use an unconditional condition), and the conditional loss is computed.

    Parameters:
        device: Device to run the evaluation on.
        model: Diffusion model (Conditioned U-Net) to evaluate.
        iterator: Data iterator yielding (data, attributes).
        BATCH_SIZE (int): Batch size.
        timesteps (int): Total number of diffusion timesteps.
        unetConfig: U-Net configuration (used for consistency).
        encodes2embeddings_mapping: Mapping from attributes to condition embeddings.
        uncondition_rate (float): Probability of using the unconditional condition.
        unconditional_condition (tensor): Unconditional condition tensor.

    Returns:
        initial_loss (float): Average loss over 500 iterations.
    """
    model.to(device)
    model.eval()
    eva_loss = []
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, _, _ = get_beta_schedule(timesteps)
    for i in xrange(500):
        data, attributes = next(iter(iterator))
        data = data.to(device)

        # Map each attribute to its corresponding condition embedding
        conditions = [encodes2embeddings_mapping[create_key(attribute)] for attribute in attributes]
        selected_conditions = [
            unconditional_condition if random.random() < uncondition_rate else random.choice(conditions_of_one_sample)
            for conditions_of_one_sample in conditions]

        selected_conditions = torch.stack(selected_conditions).float().to(device)

        # Randomly choose timesteps for the diffusion process
        t = torch.randint(0, timesteps, (BATCH_SIZE,), device=device).long()
        loss = conditional_p_losses(model, data, t, selected_conditions, loss_type="huber",
                                    sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                                    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod)

        eva_loss.append(loss.item())
    initial_loss = np.mean(eva_loss)
    return initial_loss


def get_diffusion_model(model_Config, load_pretrain=False, model_name=None, device="cpu"):
    """
    Initialize and return the diffusion model (Conditioned U-Net).

    Parameters:
        model_Config (dict): Configuration dictionary for the U-Net model.
        load_pretrain (bool): Whether to load pretrained weights.
        model_name (str, optional): Model name used for loading pretrained weights.
        device (str): Device to load the model on.

    Returns:
        UNet: Initialized diffusion model.
    """
    UNet = ConditionedUnet(**model_Config)
    print(f"Model intialized, size: {sum(p.numel() for p in UNet.parameters() if p.requires_grad)}")
    UNet.to(device)

    if load_pretrain:
        print(f"Loading weights from models/{model_name}_UNet.pth")
        checkpoint = torch.load(f'models/{model_name}_UNet.pth', map_location=device)
        UNet.load_state_dict(checkpoint['model_state_dict'])
    UNet.eval()
    return UNet


def train_diffusion_model(VAE, text_encoder, CLAP_tokenizer, timbre_encoder, device, init_model_name, unetConfig, BATCH_SIZE, timesteps, lr, max_iter, iterator, load_pretrain,
                          encodes2embeddings_mapping, uncondition_rate, unconditional_condition, save_steps=5000, init_loss=None, save_model_name=None,
                            n_IS_batches=50):
    """
    Train the diffusion model (Conditioned U-Net) using the provided data iterator and configurations.

    Parameters:
        VAE: Variational Autoencoder model.
        text_encoder: Text encoder model.
        CLAP_tokenizer: Tokenizer for CLAP.
        timbre_encoder: Timbre encoder model.
        device: Device for training (e.g., "cuda" or "cpu").
        init_model_name (str): Initial model name for checkpointing.
        unetConfig (dict): Configuration for the U-Net model.
        BATCH_SIZE (int): Batch size.
        timesteps (int): Total diffusion timesteps.
        lr (float): Learning rate.
        max_iter (int): Maximum training iterations.
        iterator: Data iterator yielding training batches.
        load_pretrain (bool): Whether to load pretrained weights.
        encodes2embeddings_mapping: Mapping from attribute keys to embeddings.
        uncondition_rate (float): Probability of applying the unconditional condition.
        unconditional_condition (tensor): Unconditional condition tensor.
        save_steps (int): Frequency (in steps) to save checkpoints.
        init_loss (float, optional): Initial loss value.
        save_model_name (str, optional): Name for saving the model.
        n_IS_batches (int): Number of batches for inception score evaluation.

    Returns:
        model: Trained diffusion model.
        optimizer: Optimizer used during training.
    """
    if save_model_name is None:
        save_model_name = init_model_name

    def save_model_hyperparameter(model_name, unetConfig, BATCH_SIZE, lr, model_size, current_iter, current_loss):
        """
        Save model hyperparameters to a JSON file for record keeping.

        Parameters:
            model_name (str): Model name.
            unetConfig (dict): U-Net configuration.
            BATCH_SIZE (int): Training batch size.
            lr (float): Learning rate.
            model_size (int): Number of trainable parameters.
            current_iter (int): Current training iteration.
            current_loss (float): Current loss value.
        """
        model_hyperparameter = unetConfig
        model_hyperparameter["BATCH_SIZE"] = BATCH_SIZE
        model_hyperparameter["lr"] = lr
        model_hyperparameter["model_size"] = model_size
        model_hyperparameter["current_iter"] = current_iter
        model_hyperparameter["current_loss"] = current_loss
        with open(f"models/hyperparameters/{model_name}_UNet.json", "w") as json_file:
            json.dump(model_hyperparameter, json_file, ensure_ascii=False, indent=4)

    # Initialize the U-Net model with the specified configuration
    model = ConditionedUnet(**unetConfig)
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {model_size}")
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, amsgrad=False)

    if load_pretrain:
        print(f"Loading weights from models/{init_model_name}_UNet.pt")
        checkpoint = torch.load(f'models/{init_model_name}_UNet.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("Model initialized.")
    if max_iter == 0:
        print("Return model directly.")
        return model, optimizer

    train_loss = []
    writer = SummaryWriter(f'runs/{save_model_name}_UNet')
    # Evaluate initial loss using a validation batch if not provided
    if init_loss is None:
        previous_loss = evaluate_diffusion_model(device, model, iterator, BATCH_SIZE, timesteps, unetConfig, encodes2embeddings_mapping,
                                                 uncondition_rate, unconditional_condition)
    else:
        previous_loss = init_loss
    print(f"initial_IS: {previous_loss}")
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, _, _ = get_beta_schedule(timesteps)

    model.train()
    for i in xrange(max_iter):
        data, attributes = next(iter(iterator))
        data = data.to(device)

        # Map each attribute to its corresponding condition embedding
        conditions = [encodes2embeddings_mapping[create_key(attribute)] for attribute in attributes]
        unconditional_condition_copy = torch.tensor(unconditional_condition, dtype=torch.float32).to(device).detach()
        selected_conditions = [unconditional_condition_copy if random.random() < uncondition_rate else random.choice(
            conditions_of_one_sample) for conditions_of_one_sample in conditions]

        selected_conditions = torch.stack(selected_conditions).float().to(device)

        optimizer.zero_grad()

        # Randomly select diffusion timesteps for the current batch
        t = torch.randint(0, timesteps, (BATCH_SIZE,), device=device).long()
        loss = conditional_p_losses(model, data, t, selected_conditions, loss_type="huber",
                                    sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                                    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        # Retrieve the current training step from the optimizer state
        step = int(optimizer.state_dict()['state'][list(optimizer.state_dict()['state'].keys())[0]]['step'].numpy())

        if step % 100 == 0:
            print('%d step' % (step))

        if step % save_steps == 0:
            current_loss = np.mean(train_loss[-save_steps:])
            print(f"current_loss = {current_loss}")
            # Save the latest model checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'models/{save_model_name}_UNet.pth')
            save_model_hyperparameter(save_model_name, unetConfig, BATCH_SIZE, lr, model_size, step, current_loss)


        if step % 20000 == 0:
            # Evaluate the model using the inception score
            current_IS = get_inception_score(device, model, VAE, text_encoder, CLAP_tokenizer, timbre_encoder, n_IS_batches,
                                     positive_prompts="", negative_prompts="", CFG=1, sample_steps=20, task="STFT")
            print('current_IS: %.5f' % current_IS)
            current_loss = np.mean(train_loss[-save_steps:])

            writer.add_scalar(f"current_IS", current_IS, step)

            # Save a historical checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'models/history/{save_model_name}_{step}_UNet.pth')
            save_model_hyperparameter(save_model_name, unetConfig, BATCH_SIZE, lr, model_size, step, current_loss)

    return model, optimizer


