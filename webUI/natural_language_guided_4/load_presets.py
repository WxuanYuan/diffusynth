import os

import librosa
import mido
import numpy as np
import torch

from tools import read_wav_to_numpy, pad_STFT, encode_stft
from webUI.natural_language_guided_4.gradio_webUI import GradioWebUI
from webUI.natural_language_guided_4.utils import InputBatch2Encode_STFT


def load_presets(gradioWebUI: GradioWebUI):
    """
    Load preset virtual instruments and MIDI files.

    This function loads preset instruments from WAV files and converts them into latent
    representations using the VAE encoder and quantizer. It also loads preset MIDI files.

    Args:
        gradioWebUI (GradioWebUI): An instance of GradioWebUI containing model configurations.

    Returns:
        tuple: A tuple containing:
            - virtual_instruments (dict): A dictionary mapping preset instrument names to their data.
            - midis (dict): A dictionary mapping preset MIDI names to their mido.MidiFile objects.
    """
    # Load configurations from the GradioWebUI object.
    uNet = gradioWebUI.uNet
    freq_resolution, time_resolution = gradioWebUI.freq_resolution, gradioWebUI.time_resolution
    VAE_scale = gradioWebUI.VAE_scale
    height = int(freq_resolution / VAE_scale)
    width = int(time_resolution / VAE_scale)
    channels = gradioWebUI.channels

    timesteps = gradioWebUI.timesteps
    VAE_quantizer = gradioWebUI.VAE_quantizer
    VAE_encoder = gradioWebUI.VAE_encoder
    VAE_decoder = gradioWebUI.VAE_decoder
    CLAP = gradioWebUI.CLAP
    CLAP_tokenizer = gradioWebUI.CLAP_tokenizer
    device = gradioWebUI.device
    squared = gradioWebUI.squared
    sample_rate = gradioWebUI.sample_rate
    noise_strategy = gradioWebUI.noise_strategy

    def add_preset_instruments(virtual_instruments, instrument_name):
        """
        Add a preset instrument to the virtual instruments dictionary.

        This function reads a WAV file corresponding to the preset instrument,
        computes its STFT, pads it, encodes the STFT, and then extracts latent
        representations using the VAE encoder and quantizer.

        Args:
            virtual_instruments (dict): The dictionary to be updated with the new instrument.
            instrument_name (str): The name of the instrument (e.g., "ax", "organ", etc.)

        Returns:
            dict: The updated virtual_instruments dictionary with the new preset instrument added.
        """
        # Construct the path for the preset instrument WAV file.
        instruments_path = os.path.join("webUI", "presets", "instruments", f"{instrument_name}.wav")
        # Read the WAV file into a numpy array.
        sample_rate, origin_audio = read_wav_to_numpy(instruments_path)

        # Compute the Short-Time Fourier Transform (STFT) of the audio.
        D = librosa.stft(origin_audio, n_fft=1024, hop_length=256, win_length=1024)
        # Pad the STFT to match expected dimensions.
        padded_D = pad_STFT(D)
        # Encode the padded STFT.
        encoded_D = encode_stft(padded_D)

        # Convert the encoded STFT into a torch tensor and repeat it along the batch dimension (batch size = 1).
        origin_spectrogram_batch_tensor = torch.from_numpy(
            np.repeat(encoded_D[np.newaxis, :, :, :], 1, axis=0)
        ).float().to(device)

        # Obtain latent representations using the VAE encoder and quantizer.
        origin_flipped_log_spectrums, origin_flipped_phases, origin_signals, origin_latent_representations, quantized_origin_latent_representations = InputBatch2Encode_STFT(
            VAE_encoder,
            origin_spectrogram_batch_tensor,
            resolution=(512, width * VAE_scale),
            quantizer=VAE_quantizer,
            squared=squared
        )

        # Create a dictionary for the preset instrument.
        virtual_instrument = {
            "latent_representation": origin_latent_representations[0].to("cpu").detach().numpy(),
            "quantized_latent_representation": quantized_origin_latent_representations[0].to("cpu").detach().numpy(),
            "sampler": "ddim",
            "signal": (sample_rate, origin_audio),
            "spectrogram_gradio_image": origin_flipped_log_spectrums[0],
            "phase_gradio_image": origin_flipped_phases[0]
        }
        # Store the instrument in the dictionary with a key prefixed by "preset_".
        virtual_instruments[f"preset_{instrument_name}"] = virtual_instrument
        return virtual_instruments

    # Initialize an empty dictionary to store preset instruments.
    virtual_instruments = {}
    # List of preset instrument names.
    preset_instrument_names = ["ax", "electronic_sound", "organ", "synth_lead", "keyboard", "string"]
    # For each preset instrument, add it to the virtual_instruments dictionary.
    for preset_instrument_name in preset_instrument_names:
        virtual_instruments = add_preset_instruments(virtual_instruments, preset_instrument_name)

    def load_midi_files():
        """
        Load preset MIDI files from disk.

        Returns:
            dict: A dictionary mapping preset MIDI file names to their mido.MidiFile objects.
        """
        midis_dict = {}
        # List of preset MIDI file names.
        midi_file_names = ["Ode_to_Joy_Easy_variation", "Air_on_the_G_String", "Canon_in_D"]

        # For each MIDI file, construct its path and load it using mido.
        for midi_file_name in midi_file_names:
            midi_path = os.path.join("webUI", "presets", "midis", f"{midi_file_name}.mid")
            mid = mido.MidiFile(midi_path)
            midis_dict[midi_file_name] = mid

        return midis_dict

    # Load the preset MIDI files.
    midis = load_midi_files()

    # Return the dictionary of preset virtual instruments and MIDI files.
    return virtual_instruments, midis
