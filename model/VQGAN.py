import json
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from six.moves import xrange
from einops import rearrange
from torchvision import models


def Normalize(in_channels, num_groups=32, norm_type="groupnorm"):
    """Normalization layer"""

    if norm_type == "batchnorm":
        return torch.nn.BatchNorm2d(in_channels)
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def nonlinearity(x, act_type="relu"):
    """Nonlinear activation function"""

    if act_type == "relu":
        return F.relu(x)
    else:
        # swish
        return x * torch.sigmoid(x)


class VectorQuantizer(nn.Module):
    """Vector quantization layer"""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input BCHW -> (BHW)C
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances (input-embedding)^2
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding (one-hot-encoding matrix)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        min_encodings, min_encoding_indices = None, None
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, (perplexity, min_encodings, min_encoding_indices)


class VectorQuantizerEMA(nn.Module):
    """Vector quantization layer based on exponential moving average"""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        min_encodings, min_encoding_indices = None, None
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, (perplexity, min_encodings, min_encoding_indices)


class DownSample(nn.Module):
    """DownSample layer"""

    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self._conv2d = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=4,
                                 stride=2, padding=1)

    def forward(self, x):
        return self._conv2d(x)


class UpSample(nn.Module):
    """UpSample layer"""

    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self._conv2d = nn.ConvTranspose2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=4,
                                          stride=2, padding=1)

    def forward(self, x):
        return self._conv2d(x)


class ResnetBlock(nn.Module):
    """ResnetBlock is a combination of non-linearity, convolution, and normalization"""

    def __init__(self, *, in_channels, out_channels=None, double_conv=False, conv_shortcut=False,
                 dropout=0.0, temb_channels=512, norm_type="groupnorm", act_type="relu", num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.act_type = act_type

        self.norm1 = Normalize(in_channels, norm_type=norm_type, num_groups=num_groups)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)

        self.double_conv = double_conv
        if self.double_conv:
            self.norm2 = Normalize(out_channels, norm_type=norm_type, num_groups=num_groups)
            self.dropout = torch.nn.Dropout(dropout)
            self.conv2 = torch.nn.Conv2d(out_channels,
                                         out_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h, act_type=self.act_type)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb, act_type=self.act_type))[:, :, None, None]

        if self.double_conv:
            h = self.norm2(h)
            h = nonlinearity(h, act_type=self.act_type)
            h = self.dropout(h)
            h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class LinearAttention(nn.Module):
    """Efficient attention block based on <https://proceedings.mlr.press/v119/katharopoulos20a.html>"""

    def __init__(self, dim, heads=4, dim_head=32, with_skip=True):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

        self.with_skip = with_skip
        if self.with_skip:
            self.nin_shortcut = torch.nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)

        if self.with_skip:
            return self.to_out(out) + self.nin_shortcut(x)
        return self.to_out(out)


class Encoder(nn.Module):
    """The encoder, consisting of alternating stacks of ResNet blocks, efficient attention modules, and downsampling layers."""

    def __init__(self, in_channels, hidden_channels, embedding_dim, block_depth=2,
                 attn_pos=None, attn_with_skip=True, norm_type="groupnorm", act_type="relu", num_groups=32):
        super(Encoder, self).__init__()

        if attn_pos is None:
            attn_pos = []
        self._layers = nn.ModuleList([DownSample(in_channels, hidden_channels[0])])
        current_channel = hidden_channels[0]

        for i in range(1, len(hidden_channels)):
            for _ in range(block_depth - 1):
                self._layers.append(ResnetBlock(in_channels=current_channel,
                                                out_channels=current_channel,
                                                double_conv=False,
                                                conv_shortcut=False,
                                                norm_type=norm_type,
                                                act_type=act_type,
                                                num_groups=num_groups))
                if current_channel in attn_pos:
                    self._layers.append(LinearAttention(current_channel, 1, 32, attn_with_skip))

            self._layers.append(Normalize(current_channel, norm_type=norm_type, num_groups=num_groups))
            self._layers.append(nn.ReLU())
            self._layers.append(DownSample(current_channel, hidden_channels[i]))
            current_channel = hidden_channels[i]

        for _ in range(block_depth - 1):
            self._layers.append(ResnetBlock(in_channels=current_channel,
                                            out_channels=current_channel,
                                            double_conv=False,
                                            conv_shortcut=False,
                                            norm_type=norm_type,
                                            act_type=act_type,
                                            num_groups=num_groups))
            if current_channel in attn_pos:
                self._layers.append(LinearAttention(current_channel, 1, 32, attn_with_skip))

        # Conv1x1: hidden_channels[-1] -> embedding_dim
        self._layers.append(Normalize(current_channel, norm_type=norm_type, num_groups=num_groups))
        self._layers.append(nn.ReLU())
        self._layers.append(nn.Conv2d(in_channels=current_channel,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1))

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    """The decoder, consisting of alternating stacks of ResNet blocks, efficient attention modules, and upsampling layers."""

    def __init__(self, embedding_dim, hidden_channels, out_channels, block_depth=2,
                 attn_pos=None, attn_with_skip=True, norm_type="groupnorm", act_type="relu",
                                                num_groups=32):
        super(Decoder, self).__init__()

        if attn_pos is None:
            attn_pos = []
        reversed_hidden_channels = list(reversed(hidden_channels))

        # Conv1x1: hidden_channels[-1] -> embedding_dim
        self._layers = nn.ModuleList([nn.Conv2d(in_channels=embedding_dim,
                                                out_channels=reversed_hidden_channels[0],
                                                kernel_size=1, stride=1, bias=False)])

        current_channel = reversed_hidden_channels[0]

        for _ in range(block_depth - 1):
            if current_channel in attn_pos:
                self._layers.append(LinearAttention(current_channel, 1, 32, attn_with_skip))
            self._layers.append(ResnetBlock(in_channels=current_channel,
                                            out_channels=current_channel,
                                            double_conv=False,
                                            conv_shortcut=False,
                                            norm_type=norm_type,
                                            act_type=act_type,
                                            num_groups=num_groups))

        for i in range(1, len(reversed_hidden_channels)):
            self._layers.append(Normalize(current_channel, norm_type=norm_type, num_groups=num_groups))
            self._layers.append(nn.ReLU())
            self._layers.append(UpSample(current_channel, reversed_hidden_channels[i]))
            current_channel = reversed_hidden_channels[i]

            for _ in range(block_depth - 1):
                if current_channel in attn_pos:
                    self._layers.append(LinearAttention(current_channel, 1, 32, attn_with_skip))
                self._layers.append(ResnetBlock(in_channels=current_channel,
                                                out_channels=current_channel,
                                                double_conv=False,
                                                conv_shortcut=False,
                                                norm_type=norm_type,
                                                act_type=act_type,
                                                num_groups=num_groups))

        self._layers.append(Normalize(current_channel, norm_type=norm_type, num_groups=num_groups))
        self._layers.append(nn.ReLU())
        self._layers.append(UpSample(current_channel, current_channel))

        # final layers
        self._layers.append(ResnetBlock(in_channels=current_channel,
                                        out_channels=out_channels,
                                        double_conv=False,
                                        conv_shortcut=False,
                                        norm_type=norm_type,
                                        act_type=act_type,
                                        num_groups=num_groups))


    def forward(self, x):
        for layer in self._layers:
            x = layer(x)

        log_magnitude = torch.nn.functional.softplus(x[:, 0, :, :])

        cos_phase = torch.tanh(x[:, 1, :, :])
        sin_phase = torch.tanh(x[:, 2, :, :])
        x = torch.stack([log_magnitude, cos_phase, sin_phase], dim=1)

        return x


class VQGAN_Discriminator(nn.Module):
    """The discriminator employs an 18-layer-ResNet architecture , with the first layer replaced by a 2D convolutional
    layer that accommodates spectral representation inputs and the last two layers replaced by a binary classifier
    layer."""

    def __init__(self, in_channels=1):
        super(VQGAN_Discriminator, self).__init__()
        resnet = models.resnet18(pretrained=True)

        # 修改第一层以接受单通道（黑白）图像
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 使用ResNet的特征提取部分
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # 添加判别器的额外层
        self.classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VQGAN(nn.Module):
    """The VQ-GAN model. <https://openaccess.thecvf.com/content/CVPR2021/html/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.html?ref=>"""

    def __init__(self, in_channels, hidden_channels, embedding_dim, out_channels, block_depth=2,
                 attn_pos=None, attn_with_skip=True, norm_type="groupnorm", act_type="relu",
                 num_embeddings=1024, commitment_cost=0.25, decay=0.99, num_groups=32):
        super(VQGAN, self).__init__()

        self._encoder = Encoder(in_channels, hidden_channels, embedding_dim, block_depth=block_depth,
                                attn_pos=attn_pos, attn_with_skip=attn_with_skip, norm_type=norm_type, act_type="act_type", num_groups=num_groups)

        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim, hidden_channels, out_channels, block_depth=block_depth,
                                attn_pos=attn_pos, attn_with_skip=attn_with_skip, norm_type=norm_type,
                                act_type=act_type, num_groups=num_groups)

    def forward(self, x):
        z = self._encoder(x)
        quantized, vq_loss, (perplexity, _, _) = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return vq_loss, x_recon, perplexity


class ReconstructionLoss(nn.Module):
    """
    Computes the weighted reconstruction loss for VQGAN.

    It calculates a weighted Mean Absolute Error (MAE) for the magnitude channel
    and a standard MAE for the phase channels.
    """
    def __init__(self, w1, w2, epsilon=1e-3):
        """
        Args:
            w1 (float): Weight for the magnitude loss.
            w2 (float): Weight for the phase loss.
            epsilon (float): Small constant to avoid division by zero.
        """
        super(ReconstructionLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.epsilon = epsilon

    def weighted_mae_loss(self, y_true, y_pred):
        """
        Computes weighted MAE loss to account for small true values.
        """
        # avoid divide by zero
        y_true_safe = torch.clamp(y_true, min=self.epsilon)

        # compute weighted MAE
        loss = torch.mean(torch.abs(y_pred - y_true) / y_true_safe)
        return loss

    def mae_loss(self, y_true, y_pred):
        """
        Computes standard MAE loss.
        """
        loss = torch.mean(torch.abs(y_pred - y_true))
        return loss

    def forward(self, y_pred, y_true):
        """
        Computes the overall reconstruction loss.

        Args:
            y_pred (Tensor): Predicted output.
            y_true (Tensor): Ground truth.

        Returns:
            log_magnitude_loss, phase_loss, rec_loss (tuple of Tensors): Individual and total losses.
        """
        # loss for magnitude channel
        log_magnitude_loss = self.w1 * self.weighted_mae_loss(y_pred[:, 0, :, :], y_true[:, 0, :, :])

        # loss for phase channels
        phase_loss = self.w2 * self.mae_loss(y_pred[:, 1:, :, :], y_true[:, 1:, :, :])

        # sum up
        rec_loss = log_magnitude_loss + phase_loss
        return log_magnitude_loss, phase_loss, rec_loss


def evaluate_VQGAN(model, discriminator, iterator, reconstructionLoss, adversarial_loss, trainingConfig):
    """
    Evaluate the VQGAN model by computing the overall loss on a number of batches.

    For each batch:
      - Compute the VQ loss and reconstruction.
      - Get the adversarial loss from the discriminator.
      - Combine the losses weighted by the configuration.

    Args:
        model (nn.Module): The VQGAN model.
        discriminator (nn.Module): The discriminator.
        iterator (iterable): Data iterator.
        reconstructionLoss (nn.Module): Reconstruction loss module.
        adversarial_loss (nn.Module): Adversarial loss function.
        trainingConfig (dict): Training configuration parameters.

    Returns:
        float: Average evaluation loss.
    """
    model.to(trainingConfig["device"])
    model.eval()
    train_res_error = []
    for i in xrange(100):
        data = next(iter(iterator))
        data = data.to(trainingConfig["device"])

        # true/fake labels
        real_labels = torch.ones(data.size(0), 1).to(trainingConfig["device"])

        vq_loss, data_recon, perplexity = model(data)


        fake_preds = discriminator(data_recon)
        adver_loss = adversarial_loss(fake_preds, real_labels)

        log_magnitude_loss, phase_loss, rec_loss = reconstructionLoss(data_recon, data)
        loss = rec_loss + trainingConfig["vq_weight"] * vq_loss + trainingConfig["adver_weight"] * adver_loss

        train_res_error.append(loss.item())
    initial_loss = np.mean(train_res_error)
    return initial_loss


def get_VQGAN(model_Config, load_pretrain=False, model_name=None, device="cpu"):
    """
    Initialize the VQGAN model and optionally load pretrained weights.

    Args:
        model_Config (dict): Configuration for VQGAN.
        load_pretrain (bool): Whether to load pretrained weights.
        model_name (str): Checkpoint name.
        device (str): Device to load the model on.

    Returns:
        nn.Module: The VQGAN model.
    """
    VQVAE = VQGAN(**model_Config)
    print(f"Model intialized, size: {sum(p.numel() for p in VQVAE.parameters() if p.requires_grad)}")
    VQVAE.to(device)

    if load_pretrain:
        print(f"Loading weights from models/{model_name}_imageVQVAE.pth")
        checkpoint = torch.load(f'models/{model_name}_imageVQVAE.pth', map_location=device)
        VQVAE.load_state_dict(checkpoint['model_state_dict'])
    VQVAE.eval()
    return VQVAE


def train_VQGAN(model_Config, trainingConfig, iterator):
    """
    Train the VQGAN model using the provided configuration and training parameters.

    The training process includes:
      - Initializing the VQGAN model and discriminator.
      - Updating the discriminator and generator (VQVAE) alternately.
      - Logging reconstruction, VQ, and adversarial losses.
      - Saving checkpoints when improvements are observed.

    Args:
        model_Config (dict): Configuration for the VQGAN model.
        trainingConfig (dict): Training configuration (lr, weights, etc.).
        iterator (iterable): Data iterator for training.

    Returns:
        nn.Module: The trained VQGAN model.
    """
    def save_model_hyperparameter(model_Config, trainingConfig, current_iter,
                                  log_magnitude_loss, phase_loss, current_perplexity, current_vq_loss,
                                  current_loss):
        model_name = trainingConfig["model_name"]
        model_hyperparameter = model_Config
        model_hyperparameter.update(trainingConfig)
        model_hyperparameter["current_iter"] = current_iter
        model_hyperparameter["log_magnitude_loss"] = log_magnitude_loss
        model_hyperparameter["phase_loss"] = phase_loss
        model_hyperparameter["erplexity"] = current_perplexity
        model_hyperparameter["vq_loss"] = current_vq_loss
        model_hyperparameter["total_loss"] = current_loss

        with open(f"models/hyperparameters/{model_name}_VQGAN_STFT.json", "w") as json_file:
            json.dump(model_hyperparameter, json_file, ensure_ascii=False, indent=4)

    # initialize VAE
    model = VQGAN(**model_Config)
    print(f"VQ_VAE size: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.to(trainingConfig["device"])

    VAE_optimizer = torch.optim.Adam(model.parameters(), lr=trainingConfig["lr"], amsgrad=False)
    model_name = trainingConfig["model_name"]

    if trainingConfig["load_pretrain"]:
        print(f"Loading weights from models/{model_name}_imageVQVAE.pth")
        checkpoint = torch.load(f'models/{model_name}_imageVQVAE.pth', map_location=trainingConfig["device"])
        model.load_state_dict(checkpoint['model_state_dict'])
        VAE_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("VAE initialized.")
    if trainingConfig["max_iter"] == 0:
        print("Return VAE directly.")
        return model

    # initialize discriminator
    discriminator = VQGAN_Discriminator(model_Config["in_channels"])
    print(f"Discriminator size: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad)}")
    discriminator.to(trainingConfig["device"])

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=trainingConfig["d_lr"], amsgrad=False)

    if trainingConfig["load_pretrain"]:
        print(f"Loading weights from models/{model_name}_imageVQVAE_discriminator.pth")
        checkpoint = torch.load(f'models/{model_name}_imageVQVAE_discriminator.pth', map_location=trainingConfig["device"])
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("Discriminator initialized.")

    # Training

    # Initialize lists to record losses.
    train_res_phase_loss, train_res_perplexity, train_res_log_magnitude_loss, train_res_vq_loss, train_res_loss = [], [], [], [], []
    train_discriminator_loss, train_adverserial_loss = [], []

    reconstructionLoss = ReconstructionLoss(w1=trainingConfig["w1"], w2=trainingConfig["w2"], epsilon=trainingConfig["threshold"])

    adversarial_loss = nn.BCEWithLogitsLoss()
    writer = SummaryWriter(f'runs/{model_name}_VQVAE_lr=1e-4')

    # Evaluate the initial loss.
    previous_lowest_loss = evaluate_VQGAN(model, discriminator, iterator,
                                          reconstructionLoss, adversarial_loss, trainingConfig)
    print(f"initial_loss: {previous_lowest_loss}")

    model.train()
    for i in xrange(trainingConfig["max_iter"]):
        data = next(iter(iterator))
        data = data.to(trainingConfig["device"])

        # Create real and fake labels for adversarial loss.
        real_labels = torch.ones(data.size(0), 1).to(trainingConfig["device"])
        fake_labels = torch.zeros(data.size(0), 1).to(trainingConfig["device"])

        # update discriminator
        discriminator_optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)

        real_preds = discriminator(data)
        fake_preds = discriminator(data_recon.detach())

        loss_real = adversarial_loss(real_preds, real_labels)
        loss_fake = adversarial_loss(fake_preds, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        discriminator_optimizer.step()


        # update VQVAE
        VAE_optimizer.zero_grad()

        fake_preds = discriminator(data_recon)
        adver_loss = adversarial_loss(fake_preds, real_labels)

        log_magnitude_loss, phase_loss, rec_loss = reconstructionLoss(data_recon, data)

        loss = rec_loss + trainingConfig["vq_weight"] * vq_loss + trainingConfig["adver_weight"] * adver_loss
        loss.backward()
        VAE_optimizer.step()

        # Record losses.
        train_discriminator_loss.append(loss_D.item())
        train_adverserial_loss.append(trainingConfig["adver_weight"] * adver_loss.item())
        train_res_log_magnitude_loss.append(log_magnitude_loss.item())
        train_res_phase_loss.append(phase_loss.item())
        train_res_perplexity.append(perplexity.item())
        train_res_vq_loss.append(trainingConfig["vq_weight"] * vq_loss.item())
        train_res_loss.append(loss.item())
        step = int(VAE_optimizer.state_dict()['state'][list(VAE_optimizer.state_dict()['state'].keys())[0]]['step'].cpu().numpy())

        save_steps = trainingConfig["save_steps"]
        if (i + 1) % 100 == 0:
            print('%d step' % (step))

        # Periodically log metrics and save checkpoints.
        if (i + 1) % save_steps == 0:
            current_discriminator_loss = np.mean(train_discriminator_loss[-save_steps:])
            current_adverserial_loss = np.mean(train_adverserial_loss[-save_steps:])
            current_log_magnitude_loss = np.mean(train_res_log_magnitude_loss[-save_steps:])
            current_phase_loss = np.mean(train_res_phase_loss[-save_steps:])
            current_perplexity = np.mean(train_res_perplexity[-save_steps:])
            current_vq_loss = np.mean(train_res_vq_loss[-save_steps:])
            current_loss = np.mean(train_res_loss[-save_steps:])

            print('discriminator_loss: %.3f' % current_discriminator_loss)
            print('adverserial_loss: %.3f' % current_adverserial_loss)
            print('log_magnitude_loss: %.3f' % current_log_magnitude_loss)
            print('phase_loss: %.3f' % current_phase_loss)
            print('perplexity: %.3f' % current_perplexity)
            print('vq_loss: %.3f' % current_vq_loss)
            print('total_loss: %.3f' % current_loss)
            writer.add_scalar(f"log_magnitude_loss", current_log_magnitude_loss, step)
            writer.add_scalar(f"phase_loss", current_phase_loss, step)
            writer.add_scalar(f"perplexity", current_perplexity, step)
            writer.add_scalar(f"vq_loss", current_vq_loss, step)
            writer.add_scalar(f"total_loss", current_loss, step)
            if current_loss < previous_lowest_loss:
                previous_lowest_loss = current_loss

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': VAE_optimizer.state_dict(),
                }, f'models/{model_name}_imageVQVAE.pth')

                torch.save({
                    'model_state_dict': discriminator.state_dict(),
                    'optimizer_state_dict': discriminator_optimizer.state_dict(),
                }, f'models/{model_name}_imageVQVAE_discriminator.pth')

                save_model_hyperparameter(model_Config, trainingConfig, step,
                                          current_log_magnitude_loss, current_phase_loss, current_perplexity, current_vq_loss,
                                          current_loss)

    return model