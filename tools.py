import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
from scipy.io import wavfile
from scipy.io.wavfile import write
import torch

k = 1e-16

def np_log10(x):
    """Safe log function with base 10."""
    numerator = np.log(x + 1e-16)
    denominator = np.log(10)
    return numerator / denominator


def sigmoid(x):
    """Safe log function with base 10."""
    s = 1 / (1 + np.exp(-x))
    return s


def inv_sigmoid(s):
    """Safe inverse sigmoid function."""
    x = np.log((s / (1 - s)) + 1e-16)
    return x


def spc_to_VAE_input(spc):
    """Restrict value range from [0, infinite] to [0, 1]. (deprecated )"""
    return spc / (1 + spc)


def VAE_out_put_to_spc(o):
    """Inverse transform of function 'spc_to_VAE_input'. (deprecated )"""
    return o / (1 - o + k)



def np_power_to_db(S, amin=1e-16, top_db=80.0):
    """Helper method for numpy data scaling. (deprecated )"""
    ref = S.max()

    log_spec = 10.0 * np_log10(np.maximum(amin, S))
    log_spec -= 10.0 * np_log10(np.maximum(amin, ref))

    log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def show_spc(spc):
    """Show a spectrogram. (deprecated )"""
    s = np.shape(spc)
    spc = np.reshape(spc, (s[0], s[1]))
    magnitude_spectrum = np.abs(spc)
    log_spectrum = np_power_to_db(magnitude_spectrum)
    plt.imshow(np.flipud(log_spectrum))
    plt.show()


def save_results(spectrogram, spectrogram_image_path, waveform_path):
    """Save the input 'spectrogram' and its waveform (reconstructed by Griffin Lim)
     to path provided by 'spectrogram_image_path' and 'waveform_path'."""
    magnitude_spectrum = np.abs(spectrogram)
    log_spc = np_power_to_db(magnitude_spectrum)
    log_spc = np.reshape(log_spc, (512, 256))
    matplotlib.pyplot.imsave(spectrogram_image_path, log_spc, vmin=-100, vmax=0,
                             origin='lower')

    # save waveform
    abs_spec = np.zeros((513, 256))
    abs_spec[:512, :] = abs_spec[:512, :] + np.sqrt(np.reshape(spectrogram, (512, 256)))
    rec_signal = librosa.griffinlim(abs_spec, n_iter=32, hop_length=256, win_length=1024)
    write(waveform_path, 16000, rec_signal)


def plot_log_spectrogram(signal: np.ndarray,
                         path: str,
                         n_fft=2048,
                         frame_length=1024,
                         frame_step=256):
    """Save spectrogram."""
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=frame_step, win_length=frame_length)
    amp = np.square(np.real(stft)) + np.square(np.imag(stft))
    magnitude_spectrum = np.abs(amp)
    log_mel = np_power_to_db(magnitude_spectrum)
    matplotlib.pyplot.imsave(path, log_mel, vmin=-100, vmax=0, origin='lower')


def visualize_feature_maps(device, model, inputs, channel_indices=[0, 3,]):
    """
    Visualize feature maps before and after quantization for given input.

    Parameters:
    - model: Your VQ-VAE model.
    - inputs: A batch of input data.
    - channel_indices: Indices of feature map channels to visualize.
    """
    model.eval()
    inputs = inputs.to(device)

    with torch.no_grad():
        z_e = model._encoder(inputs)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = model._vq_vae(z_e)

    # Assuming inputs have shape [batch_size, channels, height, width]
    batch_size = z_e.size(0)

    for idx in range(batch_size):
        fig, axs = plt.subplots(1, len(channel_indices)*2, figsize=(15, 5))

        for i, channel_idx in enumerate(channel_indices):
            # Plot encoder output
            axs[2*i].imshow(z_e[idx][channel_idx].cpu().numpy(), cmap='viridis')
            axs[2*i].set_title(f"Encoder Output - Channel {channel_idx}")

            # Plot quantized output
            axs[2*i+1].imshow(z_q[idx][channel_idx].cpu().numpy(), cmap='viridis')
            axs[2*i+1].set_title(f"Quantized Output - Channel {channel_idx}")

        plt.show()


def adjust_audio_length(audio, desired_length, original_sample_rate, target_sample_rate):
    """
    Adjust the audio length to the desired length and resample to target sample rate.

    Parameters:
    - audio (np.array): The input audio signal
    - desired_length (int): The desired length of the output audio
    - original_sample_rate (int): The original sample rate of the audio
    - target_sample_rate (int): The target sample rate for the output audio

    Returns:
    - np.array: The adjusted and resampled audio
    """

    if not (original_sample_rate == target_sample_rate):
        audio = librosa.core.resample(audio, orig_sr=original_sample_rate, target_sr=target_sample_rate)

    if len(audio) > desired_length:
        return audio[:desired_length]

    elif len(audio) < desired_length:
        padded_audio = np.zeros(desired_length)
        padded_audio[:len(audio)] = audio
        return padded_audio
    else:
        return audio


def safe_int(s, default=0):
    try:
        return int(s)
    except ValueError:
        return default


def pad_spectrogram(D):
    """Resize spectrogram to (512, 256). (deprecated )"""
    D = D[1:, :]

    padding_length = 256 - D.shape[1]
    D_padded = np.pad(D, ((0, 0), (0, padding_length)), 'constant')
    return D_padded


def pad_STFT(D, time_resolution=256):
    """Resize spectral matrix by padding and cropping"""
    D = D[1:, :]

    if time_resolution is None:
        return D

    padding_length = time_resolution - D.shape[1]
    if padding_length > 0:
        D_padded = np.pad(D, ((0, 0), (0, padding_length)), 'constant')
        return D_padded
    else:
        return D


def depad_STFT(D_padded):
    """Inverse function of 'pad_STFT'"""
    zero_row = np.zeros((1, D_padded.shape[1]))

    D_restored = np.concatenate([zero_row, D_padded], axis=0)

    return D_restored


def nnData2Audio(spectrogram_batch, resolution=(512, 256), squared=False):
    """Transform batch of numpy spectrogram into signals and encodings."""
    # Todo: remove resolution hard-coding
    frequency_resolution, time_resolution = resolution

    if isinstance(spectrogram_batch, torch.Tensor):
        spectrogram_batch = spectrogram_batch.to("cpu").detach().numpy()

    origin_signals = []
    for spectrogram in spectrogram_batch:
        spc = VAE_out_put_to_spc(spectrogram)

        # get_audio
        abs_spec = np.zeros((frequency_resolution+1, time_resolution))

        if squared:
            abs_spec[1:, :] = abs_spec[1:, :] + np.sqrt(np.reshape(spc, (frequency_resolution, time_resolution)))
        else:
            abs_spec[1:, :] = abs_spec[1:, :] + np.reshape(spc, (frequency_resolution, time_resolution))

        origin_signal = librosa.griffinlim(abs_spec, n_iter=32, hop_length=256, win_length=1024)
        origin_signals.append(origin_signal)

    return origin_signals


def amp_to_audio(amp, n_iter=50):
    """The Griffin-Lim algorithm."""
    y_reconstructed = librosa.griffinlim(amp, n_iter=n_iter, hop_length=256, win_length=1024)
    return y_reconstructed


def rescale(amp, method="log1p"):
    """Rescale function."""
    if method == "log1p":
        return np.log1p(amp)
    elif method == "NormalizedLogisticCompression":
        return amp / (1.0 + amp)
    else:
        raise NotImplementedError()


def unrescale(scaled_amp, method="NormalizedLogisticCompression"):
    """Inverse function of 'rescale'"""
    if method == "log1p":
        return np.expm1(scaled_amp)
    elif method == "NormalizedLogisticCompression":
        return scaled_amp / (1.0 - scaled_amp + 1e-10)
    else:
        raise NotImplementedError()


def create_key(attributes):
    """Create unique key for each multi-label."""
    qualities_str = ''.join(map(str, attributes["qualities"]))
    instrument_source_str = attributes["instrument_source_str"]
    instrument_family = attributes["instrument_family_str"]
    key = f"{instrument_source_str}_{instrument_family}_{qualities_str}"
    return key


def merge_dictionaries(dicts):
    """Merge dictionaries."""
    merged_dict = {}
    for dictionary in dicts:
        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key] += value
            else:
                merged_dict[key] = value
    return merged_dict


def adsr_envelope(signal, sample_rate, duration, attack_time, decay_time, sustain_level, release_time):
    """
    Apply an ADSR envelope to an audio signal.

    :param signal: The original audio signal (numpy array).
    :param sample_rate: The sample rate of the audio signal.
    :param attack_time: Attack time in seconds.
    :param decay_time: Decay time in seconds.
    :param sustain_level: Sustain level as a fraction of the peak (0 to 1).
    :param release_time: Release time in seconds.
    :return: The audio signal with the ADSR envelope applied.
    """
    # Calculate the number of samples for each ADSR phase
    duration_samples = int(duration * sample_rate)

    # assert (duration_samples + int(1.0 * sample_rate)) <= len(signal), "(duration_samples + sample_rate) > len(signal)"
    assert release_time <= 1.0, "release_time > 1.0"

    attack_samples = int(attack_time * sample_rate)
    decay_samples = int(decay_time * sample_rate)
    release_samples = int(release_time * sample_rate)
    sustain_samples = max(0, duration_samples - attack_samples - decay_samples)

    # Create ADSR envelope
    attack_env = np.linspace(0, 1, attack_samples)
    decay_env = np.linspace(1, sustain_level, decay_samples)
    sustain_env = np.full(sustain_samples, sustain_level)
    release_env = np.linspace(sustain_level, 0, release_samples)
    release_env_expand = np.zeros(int(1.0 * sample_rate))
    release_env_expand[:len(release_env)] = release_env

    # Concatenate all phases to create the complete envelope
    envelope = np.concatenate([attack_env, decay_env, sustain_env, release_env_expand])

    # Apply the envelope to the signal
    if len(envelope) <= len(signal):
        applied_signal = signal[:len(envelope)] * envelope
    else:
        signal_expanded = np.zeros(len(envelope))
        signal_expanded[:len(signal)] = signal
        applied_signal = signal_expanded * envelope

    return applied_signal


def rms_normalize(audio, target_rms=0.1):
    """Normalize the RMS value."""
    current_rms = np.sqrt(np.mean(audio**2))
    scaling_factor = target_rms / current_rms
    normalized_audio = audio * scaling_factor
    return normalized_audio


def encode_stft(D):
    """'STFT+' function that transform spectral matrix into spectral representation."""
    magnitude = np.abs(D)
    phase = np.angle(D)

    log_magnitude = np.log1p(magnitude)

    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)

    encoded_D = np.stack([log_magnitude, cos_phase, sin_phase], axis=0)
    return encoded_D


def decode_stft(encoded_D):
    """'ISTFT+' function that reconstructs spectral matrix from spectral representation."""
    log_magnitude = encoded_D[0, ...]
    cos_phase = encoded_D[1, ...]
    sin_phase = encoded_D[2, ...]

    magnitude = np.expm1(log_magnitude)

    phase = np.arctan2(sin_phase, cos_phase)

    D = magnitude * (np.cos(phase) + 1j * np.sin(phase))
    return D


def read_wav_to_numpy(file_path):
    """
    读取 WAV 文件并返回采样率和音频数据 (NumPy 数组)

    参数:
    file_path (str): WAV 文件的路径

    返回:
    tuple: 采样率 (int) 和音频数据 (NumPy array)
    """
    # 使用 scipy.io.wavfile 读取 wav 文件
    sample_rate, data = wavfile.read(file_path)
    data = data/np.max(np.abs(data))

    return sample_rate, data


