import json
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tools import create_key


class TimbreEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, num_instrument_classes, num_instrument_family_classes, num_velocity_classes, num_qualities, num_layers=1):
        """
        Initialize the TimbreEncoder.

        The model projects the input features via a linear layer,
        then processes them with an LSTM, and finally produces outputs
        for multiple classification tasks:
          - Instrument classification
          - Instrument family classification
          - Velocity classification
          - Qualities prediction (using sigmoid)

        Args:
            input_dim (int): Dimension of the input features.
            feature_dim (int): Dimension after the linear projection.
            hidden_dim (int): Hidden state size for the LSTM.
            num_instrument_classes (int): Number of instrument classes.
            num_instrument_family_classes (int): Number of instrument family classes.
            num_velocity_classes (int): Number of velocity classes.
            num_qualities (int): Number of qualities to predict.
            num_layers (int): Number of LSTM layers.
        """

        super(TimbreEncoder, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(input_dim, feature_dim)

        # LSTM Layer
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Fully Connected Layers for classification
        self.instrument_classifier_layer = nn.Linear(hidden_dim, num_instrument_classes)
        self.instrument_family_classifier_layer = nn.Linear(hidden_dim, num_instrument_family_classes)
        self.velocity_classifier_layer = nn.Linear(hidden_dim, num_velocity_classes)
        self.qualities_classifier_layer = nn.Linear(hidden_dim, num_qualities)

        # Softmax for converting output to probabilities
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the TimbreEncoder.

        Args:
            x (Tensor): Input tensor with shape [batch_size, ?, ?, seq_len].

        Returns:
            feature (Tensor): Final LSTM features (last time step).
            instrument_logits (Tensor): Log probabilities for instrument classification.
            instrument_family_logits (Tensor): Log probabilities for instrument family classification.
            velocity_logits (Tensor): Log probabilities for velocity classification.
            qualities (Tensor): Sigmoid output for qualities.
        """
        # # Merge first two dimensions
        batch_size, _, _, seq_len = x.shape
        x = x.view(batch_size, -1, seq_len)  # [batch_size, input_dim, seq_len]

        # Forward propagate LSTM
        x = x.permute(0, 2, 1)
        x = self.input_layer(x)
        feature, _ = self.lstm(x)
        feature = feature[:, -1, :]

        # Apply classification layers
        instrument_logits = self.instrument_classifier_layer(feature)
        instrument_family_logits = self.instrument_family_classifier_layer(feature)
        velocity_logits = self.velocity_classifier_layer(feature)
        qualities = self.qualities_classifier_layer(feature)

        # Apply Softmax
        instrument_logits = self.softmax(instrument_logits)
        instrument_family_logits= self.softmax(instrument_family_logits)
        velocity_logits = self.softmax(velocity_logits)
        qualities = torch.sigmoid(qualities)

        return feature, instrument_logits, instrument_family_logits, velocity_logits, qualities


def get_multiclass_acc(outputs, ground_truth):
    """
    Compute accuracy for multi-class classification.

    Args:
        outputs (Tensor): Log probabilities (output from LogSoftmax).
        ground_truth (Tensor): True class indices.

    Returns:
        float: Accuracy percentage.
    """

    _, predicted = torch.max(outputs.data, 1)
    total = ground_truth.size(0)
    correct = (predicted == ground_truth).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def get_binary_accuracy(y_pred, y_true):
    """
    Compute binary classification accuracy.

    Args:
        y_pred (Tensor): Predicted probabilities.
        y_true (Tensor): Ground truth binary labels.

    Returns:
        float: Accuracy percentage.
    """

    predictions = (y_pred > 0.5).int()

    correct_predictions = (predictions == y_true).float()

    accuracy = correct_predictions.mean()

    return accuracy.item() * 100.0


def get_timbre_encoder(model_Config, load_pretrain=False, model_name=None, device="cpu"):
    """
    Initialize the TimbreEncoder with the provided configuration.

    Optionally load pretrained weights if load_pretrain is True.

    Args:
        model_Config (dict): Configuration parameters for TimbreEncoder.
        load_pretrain (bool): Whether to load pretrained weights.
        model_name (str): Model name used to load weights.
        device (str): Device on which to load the model.

    Returns:
        nn.Module: Initialized TimbreEncoder.
    """
    timbreEncoder = TimbreEncoder(**model_Config)
    print(f"Model intialized, size: {sum(p.numel() for p in timbreEncoder.parameters() if p.requires_grad)}")
    timbreEncoder.to(device)

    if load_pretrain:
        print(f"Loading weights from models/{model_name}_timbre_encoder.pth")
        checkpoint = torch.load(f'models/{model_name}_timbre_encoder.pth', map_location=device)
        timbreEncoder.load_state_dict(checkpoint['model_state_dict'])
    timbreEncoder.eval()
    return timbreEncoder


def evaluate_timbre_encoder(device, model, iterator, nll_Loss, bce_Loss, n_sample=100):
    """
    Evaluate the TimbreEncoder over several batches.

    For each batch, extract the ground truth for each classification task,
    run the model to obtain predictions, compute the losses, and return the average loss.

    Args:
        device (str): Device on which to run evaluation.
        model (nn.Module): The TimbreEncoder model.
        iterator (iterable): Data iterator that yields (representation, attributes).
        nll_Loss: Loss function for multi-class classification.
        bce_Loss: Loss function for binary classification.
        n_sample (int): Number of batches to evaluate.

    Returns:
        float: Average evaluation loss.
    """
    model.to(device)
    model.eval()

    eva_loss = []
    for i in range(n_sample):
        representation, attributes = next(iter(iterator))

        instrument = torch.tensor([s["instrument"] for s in attributes], dtype=torch.long).to(device)
        instrument_family = torch.tensor([s["instrument_family"] for s in attributes], dtype=torch.long).to(device)
        velocity = torch.tensor([s["velocity"] for s in attributes], dtype=torch.long).to(device)
        qualities = torch.tensor([[int(char) for char in create_key(attribute)[-10:]] for attribute in attributes], dtype=torch.float32).to(device)

        _, instrument_logits, instrument_family_logits, velocity_logits, qualities_pred = model(representation.to(device))

        # compute loss
        instrument_loss = nll_Loss(instrument_logits, instrument)
        instrument_family_loss = nll_Loss(instrument_family_logits, instrument_family)
        velocity_loss = nll_Loss(velocity_logits, velocity)
        qualities_loss = bce_Loss(qualities_pred, qualities)

        loss = instrument_loss + instrument_family_loss + velocity_loss + qualities_loss

        eva_loss.append(loss.item())

    eva_loss = np.mean(eva_loss)
    return eva_loss


def train_timbre_encoder(device, model_name, timbre_encoder_Config, BATCH_SIZE, lr, max_iter, training_iterator, load_pretrain):
    """
    Train the TimbreEncoder over multiple iterations.

    The training loop:
      - Loads batches from the training iterator.
      - Extracts and processes the ground truth labels.
      - Computes the model loss (sum of individual losses for each classification task).
      - Backpropagates the loss and updates the model parameters.
      - Records loss and accuracy metrics, and logs them to TensorBoard.
      - Saves the model checkpoint if the loss improves.

    Args:
        device (str): Device to run training on.
        model_name (str): Name of the model for saving checkpoints.
        timbre_encoder_Config (dict): Configuration for the TimbreEncoder.
        BATCH_SIZE (int): Batch size.
        lr (float): Learning rate.
        max_iter (int): Maximum number of training iterations.
        training_iterator (iterable): Iterator that yields training batches.
        load_pretrain (bool): Whether to load pretrained weights.

    Returns:
        tuple: The final model and the best model found during training.
    """
    def save_model_hyperparameter(model_name, timbre_encoder_Config, BATCH_SIZE, lr, model_size, current_iter,
                                  current_loss):
        model_hyperparameter = timbre_encoder_Config
        model_hyperparameter["BATCH_SIZE"] = BATCH_SIZE
        model_hyperparameter["lr"] = lr
        model_hyperparameter["model_size"] = model_size
        model_hyperparameter["current_iter"] = current_iter
        model_hyperparameter["current_loss"] = current_loss
        with open(f"models/hyperparameters/{model_name}_timbre_encoder.json", "w") as json_file:
            json.dump(model_hyperparameter, json_file, ensure_ascii=False, indent=4)

    model = TimbreEncoder(**timbre_encoder_Config)
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size}")
    model.to(device)
    nll_Loss = torch.nn.NLLLoss()
    bce_Loss = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

    if load_pretrain:
        print(f"Loading weights from models/{model_name}_timbre_encoder.pt")
        checkpoint = torch.load(f'models/{model_name}_timbre_encoder.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("Model initialized.")
    if max_iter == 0:
        print("Return model directly.")
        return model, model

    train_loss, training_instrument_acc, training_instrument_family_acc, training_velocity_acc, training_qualities_acc = [], [], [], [], []
    writer = SummaryWriter(f'runs/{model_name}_timbre_encoder')
    current_best_model = model
    previous_lowest_loss = 100.0
    print(f"initial__loss: {previous_lowest_loss}")

    # Main training loop.
    for i in range(max_iter):
        model.train()

        # Get a batch of training data.
        representation, attributes = next(iter(training_iterator))

        # Extract ground truth labels for each classification task.
        instrument = torch.tensor([s["instrument"] for s in attributes], dtype=torch.long).to(device)
        instrument_family = torch.tensor([s["instrument_family"] for s in attributes], dtype=torch.long).to(device)
        velocity = torch.tensor([s["velocity"] for s in attributes], dtype=torch.long).to(device)
        qualities = torch.tensor([[int(char) for char in create_key(attribute)[-10:]] for attribute in attributes], dtype=torch.float32).to(device)

        optimizer.zero_grad()

        # Forward pass through the model.
        _, instrument_logits, instrument_family_logits, velocity_logits, qualities_pred = model(representation.to(device))

        # compute loss
        instrument_loss = nll_Loss(instrument_logits, instrument)
        instrument_family_loss = nll_Loss(instrument_family_logits, instrument_family)
        velocity_loss = nll_Loss(velocity_logits, velocity)
        qualities_loss = bce_Loss(qualities_pred, qualities)

        # Sum all losses.
        loss = instrument_loss + instrument_family_loss + velocity_loss + qualities_loss

        # Backpropagation and parameter update.
        loss.backward()
        optimizer.step()

        # Calculate accuracies for monitoring.
        instrument_acc = get_multiclass_acc(instrument_logits, instrument)
        instrument_family_acc = get_multiclass_acc(instrument_family_logits, instrument_family)
        velocity_acc = get_multiclass_acc(velocity_logits, velocity)
        qualities_acc = get_binary_accuracy(qualities_pred, qualities)

        # Record the loss and accuracies.
        train_loss.append(loss.item())
        training_instrument_acc.append(instrument_acc)
        training_instrument_family_acc.append(instrument_family_acc)
        training_velocity_acc.append(velocity_acc)
        training_qualities_acc.append(qualities_acc)
        step = int(optimizer.state_dict()['state'][list(optimizer.state_dict()['state'].keys())[0]]['step'].numpy())

        # Get the current step from the optimizer's state.
        if (i + 1) % 100 == 0:
            print('%d step' % (step))

        # Periodically compute average metrics and save checkpoints.
        save_steps = 500
        if (i + 1) % save_steps == 0:
            current_loss = np.mean(train_loss[-save_steps:])
            current_instrument_acc = np.mean(training_instrument_acc[-save_steps:])
            current_instrument_family_acc = np.mean(training_instrument_family_acc[-save_steps:])
            current_velocity_acc = np.mean(training_velocity_acc[-save_steps:])
            current_qualities_acc = np.mean(training_qualities_acc[-save_steps:])
            print('train_loss: %.5f' % current_loss)
            print('current_instrument_acc: %.5f' % current_instrument_acc)
            print('current_instrument_family_acc: %.5f' % current_instrument_family_acc)
            print('current_velocity_acc: %.5f' % current_velocity_acc)
            print('current_qualities_acc: %.5f' % current_qualities_acc)
            writer.add_scalar(f"train_loss", current_loss, step)
            writer.add_scalar(f"current_instrument_acc", current_instrument_acc, step)
            writer.add_scalar(f"current_instrument_family_acc", current_instrument_family_acc, step)
            writer.add_scalar(f"current_velocity_acc", current_velocity_acc, step)
            writer.add_scalar(f"current_qualities_acc", current_qualities_acc, step)

            if current_loss < previous_lowest_loss:
                previous_lowest_loss = current_loss
                current_best_model = model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'models/{model_name}_timbre_encoder.pth')
                save_model_hyperparameter(model_name, timbre_encoder_Config, BATCH_SIZE, lr, model_size, step,
                                          current_loss)

    return model, current_best_model


