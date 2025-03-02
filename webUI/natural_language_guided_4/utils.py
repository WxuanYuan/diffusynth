import librosa
import numpy as np
import torch
from PIL import Image
from tools import np_power_to_db, decode_stft, depad_STFT


def spectrogram_to_Gradio_image(spc):
    """
    Convert a spectrogram (usually in a numpy array format) into a colorful image
    suitable for display in Gradio.

    Steps:
      1. Reshape the input spectrogram to 2D (frequency_resolution x time_resolution).
      2. Compute the magnitude spectrum and convert it to a logarithmic (dB) scale.
      3. Flip the log spectrum vertically (so that lower frequencies are at the bottom).
      4. Create a 3-channel (RGB) image where two channels use the flipped log spectrum
         and one channel uses a constant value (e.g., -60 dB) for visual contrast.
      5. Rescale the values to the 0-255 range and convert the type to uint8.

    Args:
        spc (np.ndarray): Input spectrogram array.

    Returns:
        np.ndarray: A uint8 image array with shape (frequency_resolution, time_resolution, 3).
    """
    # Get frequency and time resolution from the spectrogram shape.
    frequency_resolution, time_resolution = spc.shape[-2], spc.shape[-1]
    # Reshape the spectrogram to a 2D array.
    spc = np.reshape(spc, (frequency_resolution, time_resolution))

    # Compute the magnitude spectrum and convert to dB scale.
    magnitude_spectrum = np.abs(spc)
    log_spectrum = np_power_to_db(magnitude_spectrum)
    # Flip the log spectrum vertically so that the frequency axis is inverted.
    flipped_log_spectrum = np.flipud(log_spectrum)

    # Create a 3-channel image. Two channels use the flipped log spectrum,
    # and the third channel is set to a constant value (-60 dB) to add visual contrast.
    colorful_spc = np.ones((frequency_resolution, time_resolution, 3)) * -80.0
    colorful_spc[:, :, 0] = flipped_log_spectrum
    colorful_spc[:, :, 1] = flipped_log_spectrum
    colorful_spc[:, :, 2] = np.ones((frequency_resolution, time_resolution)) * -60.0

    # Rescale the values: shift from [-80, 0] to [0, 1], then scale to [0, 255].
    rescaled = (colorful_spc + 80.0) / 80.0
    rescaled = (255.0 * rescaled).astype(np.uint8)
    return rescaled


def phase_to_Gradio_image(phase):
    """
    Convert a phase matrix into an image representation suitable for Gradio display.

    Steps:
      1. Reshape the input phase array to 2D.
      2. Flip the phase array vertically.
      3. Normalize the phase values from [-1, 1] to [0, 1].
      4. Create a 3-channel image using the normalized phase information (set the third channel to a constant).
      5. Scale the image values to the 0-255 range and convert to uint8.

    Args:
        phase (np.ndarray): Input phase matrix.

    Returns:
        np.ndarray: A uint8 image array representing the phase information.
    """
    # Get frequency and time resolution from the phase matrix.
    frequency_resolution, time_resolution = phase.shape[-2], phase.shape[-1]
    # Reshape the phase array to 2D.
    phase = np.reshape(phase, (frequency_resolution, time_resolution))

    # Flip the phase array vertically.
    flipped_phase = np.flipud(phase)
    # Normalize the phase values to be within [0, 1].
    flipped_phase = (flipped_phase + 1.0) / 2.0

    # Create a 3-channel image. Use the normalized phase for the first two channels and set the third channel to 0.2.
    colorful_spc = np.zeros((frequency_resolution, time_resolution, 3))
    colorful_spc[:, :, 0] = flipped_phase
    colorful_spc[:, :, 1] = flipped_phase
    colorful_spc[:, :, 2] = 0.2

    # Scale the image values to 0-255 and convert to uint8.
    rescaled = (255.0 * colorful_spc).astype(np.uint8)
    return rescaled


def latent_representation_to_Gradio_image(latent_representation):
    """
    Convert a latent representation (typically a tensor) into an image for display.

    Steps:
      1. Ensure the latent representation is a numpy array.
      2. Normalize each channel (assumed 4 channels) of the latent representation to [0, 255].
      3. Transpose the array dimensions so that the channels become the last dimension.
      4. Enlarge the image by repeating rows and columns (for better visualization).
      5. Flip the image vertically and convert to uint8.

    Args:
        latent_representation (torch.Tensor or np.ndarray): Input latent representation.

    Returns:
        np.ndarray: A uint8 image of the latent representation.
    """
    # Convert tensor to numpy array if necessary.
    if not isinstance(latent_representation, np.ndarray):
        latent_representation = latent_representation.to("cpu").detach().numpy()
    image = latent_representation

    def normalize_image(img):
        min_val = img.min()
        max_val = img.max()
        normalized_img = ((img - min_val) / (max_val - min_val) * 255)
        return normalized_img

    # Normalize each channel individually.
    image[0, :, :] = normalize_image(image[0, :, :])
    image[1, :, :] = normalize_image(image[1, :, :])
    image[2, :, :] = normalize_image(image[2, :, :])
    image[3, :, :] = normalize_image(image[3, :, :])
    # Transpose the dimensions from (channels, height, width) to (height, width, channels)
    image_transposed = np.transpose(image, (1, 2, 0))
    # Enlarge the image by repeating rows and columns (scale factor of 8)
    enlarged_image = np.repeat(image_transposed, 8, axis=0)
    enlarged_image = np.repeat(enlarged_image, 8, axis=1)
    # Flip the image vertically and convert to uint8.
    return np.flipud(enlarged_image).astype(np.uint8)


def InputBatch2Encode_STFT(encoder, STFT_batch, resolution=(512, 256), quantizer=None, squared=True):
    """
    Encode a batch of spectrograms using the provided encoder (and optionally a quantizer).

    Steps:
      1. Get the frequency and time resolution from the resolution parameter.
      2. Move the input STFT batch to the appropriate device.
      3. If a quantizer is provided, pass the encoded batch through it.
      4. Convert the STFT batch to a numpy array.
      5. For each STFT in the batch, decode it back to the time-frequency domain,
         depad it, and compute both the magnitude (spectrogram) and phase images.
      6. Return the list of processed spectrogram images, phase images, reconstructed signals,
         along with the latent representations.

    Args:
        encoder: The neural network encoder for spectrograms.
        STFT_batch (torch.Tensor): Batch of STFT representations.
        resolution (tuple): Desired resolution (frequency, time) for decoding.
        quantizer (optional): A VAE quantizer module; if provided, quantize the latent representations.
        squared (bool): Whether the encoder output is squared.

    Returns:
        tuple: (List of spectrogram images, list of phase images, list of reconstructed signals,
                latent_representation_batch, quantized_latent_representation_batch)
    """
    frequency_resolution, time_resolution = resolution

    device = next(encoder.parameters()).device
    # Encode the STFT batch using the encoder and, if provided, the quantizer.
    if not (quantizer is None):
        latent_representation_batch = encoder(STFT_batch.to(device))
        quantized_latent_representation_batch, loss, (_, _, _) = quantizer(latent_representation_batch)
    else:
        mu, logvar, latent_representation_batch = encoder(STFT_batch.to(device))
        quantized_latent_representation_batch = None

    # Move the STFT batch back to CPU as a numpy array.
    STFT_batch = STFT_batch.to("cpu").detach().numpy()

    origin_flipped_log_spectrums, origin_flipped_phases, origin_signals = [], [], []
    # Process each STFT in the batch.
    for STFT in STFT_batch:
        # Decode the STFT back into the time-frequency representation.
        padded_D_rec = decode_stft(STFT)
        D_rec = depad_STFT(padded_D_rec)
        spc = np.abs(D_rec)
        phase = np.angle(D_rec)

        # Convert the spectrogram and phase data into images for Gradio.
        flipped_log_spectrum = spectrogram_to_Gradio_image(spc)
        flipped_phase = phase_to_Gradio_image(phase)

        # Reconstruct the time-domain audio signal using the inverse STFT.
        rec_signal = librosa.istft(D_rec, hop_length=256, win_length=1024)

        origin_flipped_log_spectrums.append(flipped_log_spectrum)
        origin_flipped_phases.append(flipped_phase)
        origin_signals.append(rec_signal)

    return (origin_flipped_log_spectrums, origin_flipped_phases, origin_signals,
            latent_representation_batch, quantized_latent_representation_batch)


def encodeBatch2GradioOutput_STFT(decoder, latent_vector_batch, resolution=(512, 256), original_STFT_batch=None):
    """
    Decode a batch of latent vectors using the decoder and generate output images and signals.

    Steps:
      1. If the latent_vector_batch is a numpy array, convert it to a torch tensor and move to the decoder's device.
      2. Use the decoder to reconstruct the STFT batch from the latent vectors.
      3. For each reconstructed STFT, decode and depad it, compute the magnitude (spectrogram) and phase.
      4. Convert the spectrogram and phase into images.
      5. Optionally, if an original STFT batch is provided, replace the first channel of the STFT with
         that from the original and process it similarly.
      6. Return the lists of generated images and audio signals.

    Args:
        decoder: The neural network decoder.
        latent_vector_batch: Batch of latent vectors (torch.Tensor or numpy array).
        resolution (tuple): The resolution for the reconstructed STFT (frequency, time).
        original_STFT_batch (optional): A batch of original STFTs for reference.

    Returns:
        tuple: (flipped_log_spectrums, flipped_phases, rec_signals,
                flipped_log_spectrums_with_original_amp, flipped_phases_with_original_amp, rec_signals_with_original_amp)
    """
    frequency_resolution, time_resolution = resolution

    # Convert latent_vector_batch to a torch tensor if necessary.
    if isinstance(latent_vector_batch, np.ndarray):
        latent_vector_batch = torch.from_numpy(latent_vector_batch).to(next(decoder.parameters()).device)

    # Use the decoder to obtain the reconstruction batch.
    reconstruction_batch = decoder(latent_vector_batch).to("cpu").detach().numpy()

    flipped_log_spectrums, flipped_phases, rec_signals = [], [], []
    flipped_log_spectrums_with_original_amp, flipped_phases_with_original_amp, rec_signals_with_original_amp = [], [], []

    for index, STFT in enumerate(reconstruction_batch):
        # Decode and depad the reconstructed STFT.
        padded_D_rec = decode_stft(STFT)
        D_rec = depad_STFT(padded_D_rec)
        spc = np.abs(D_rec)
        phase = np.angle(D_rec)

        # Generate images from the spectrogram and phase.
        flipped_log_spectrum = spectrogram_to_Gradio_image(spc)
        flipped_phase = phase_to_Gradio_image(phase)

        # Reconstruct the time-domain audio signal.
        rec_signal = librosa.istft(D_rec, hop_length=256, win_length=1024)

        flipped_log_spectrums.append(flipped_log_spectrum)
        flipped_phases.append(flipped_phase)
        rec_signals.append(rec_signal)

        ##########################################
        # Optionally process the original amplitude.
        if original_STFT_batch is not None:
            # Replace the first channel with the original amplitude.
            STFT[0, :, :] = original_STFT_batch[index, 0, :, :]

            padded_D_rec = decode_stft(STFT)
            D_rec = depad_STFT(padded_D_rec)
            spc = np.abs(D_rec)
            phase = np.angle(D_rec)

            flipped_log_spectrum = spectrogram_to_Gradio_image(spc)
            flipped_phase = phase_to_Gradio_image(phase)
            rec_signal = librosa.istft(D_rec, hop_length=256, win_length=1024)

            flipped_log_spectrums_with_original_amp.append(flipped_log_spectrum)
            flipped_phases_with_original_amp.append(flipped_phase)
            rec_signals_with_original_amp.append(rec_signal)

    return (flipped_log_spectrums, flipped_phases, rec_signals,
            flipped_log_spectrums_with_original_amp, flipped_phases_with_original_amp, rec_signals_with_original_amp)


def add_instrument(source_dict, virtual_instruments_dict, virtual_instrument_name, sample_index):
    """
    Add a generated sound as a virtual instrument to the collection.

    The instrument is stored with its latent representations, quantized latent representations,
    sampler, generated audio signal, and corresponding spectrogram and phase images.

    Args:
        source_dict (dict): Dictionary containing generated outputs (e.g., from a diffusion process).
        virtual_instruments_dict (dict): Dictionary of existing virtual instruments.
        virtual_instrument_name (str): The key/name for the new instrument.
        sample_index (int): Index of the sample in the source_dict to use.

    Returns:
        dict: Updated virtual_instruments_dict.
    """
    virtual_instruments = virtual_instruments_dict["virtual_instruments"]
    virtual_instrument = {
        "latent_representation": source_dict["latent_representations"][sample_index],
        "quantized_latent_representation": source_dict["quantized_latent_representations"][sample_index],
        "sampler": source_dict["sampler"],
        "signal": source_dict["new_sound_rec_signals_gradio"][sample_index],
        "spectrogram_gradio_image": source_dict["new_sound_spectrogram_gradio_images"][sample_index],
        "phase_gradio_image": source_dict["new_sound_phase_gradio_images"][sample_index]
    }
    virtual_instruments[virtual_instrument_name] = virtual_instrument
    virtual_instruments_dict["virtual_instruments"] = virtual_instruments
    return virtual_instruments_dict


def resize_image_to_aspect_ratio(image_data, aspect_ratio_width, aspect_ratio_height):
    """
    Resize an image (numpy array) to a given aspect ratio while preserving the input format.

    Args:
        image_data (np.ndarray): Input image data with shape (height, width, 3).
        aspect_ratio_width (int): Desired width ratio.
        aspect_ratio_height (int): Desired height ratio.

    Returns:
        np.ndarray: The resized image as a numpy array.
    """
    # Get the current image dimensions.
    original_height, original_width, channels = image_data.shape

    # Calculate the current aspect ratio.
    current_aspect_ratio = original_width / original_height
    # Calculate the target aspect ratio.
    target_aspect_ratio = aspect_ratio_width / aspect_ratio_height

    # Determine whether to stretch width or height.
    if current_aspect_ratio > target_aspect_ratio:
        # If the current ratio is larger, height needs to be increased.
        new_width = original_width
        new_height = int(new_width / target_aspect_ratio)
    else:
        # Otherwise, width needs to be increased.
        new_height = original_height
        new_width = int(new_height * target_aspect_ratio)

    # Convert the numpy array to a PIL Image.
    image = Image.fromarray(image_data.astype('uint8'))
    # Resize using LANCZOS filter for high-quality downsampling.
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    # Convert back to a numpy array.
    resized_image_data = np.array(resized_image)

    return resized_image_data


def average_np_arrays(arr_list):
    """
    Compute the element-wise average of a list of numpy arrays.

    Args:
        arr_list (list): List of numpy arrays to average.

    Returns:
        np.ndarray: The averaged numpy array.

    Raises:
        ValueError: If the input list is empty.
    """
    if not arr_list:
        raise ValueError("Input list cannot be empty")

    # Stack arrays along a new axis and compute the mean.
    stacked_arrays = np.stack(arr_list, axis=0)
    avg_array = np.mean(stacked_arrays, axis=0)
    return avg_array
