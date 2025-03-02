import librosa
import numpy as np
import torch
import gradio as gr
from scipy.ndimage import zoom

# Import required modules for sampling and STFT processing
from model.DiffSynthSampler import DiffSynthSampler
from tools import adjust_audio_length, safe_int, pad_STFT, encode_stft
from webUI.natural_language_guided_4.utils import (
    latent_representation_to_Gradio_image,
    InputBatch2Encode_STFT,
    encodeBatch2GradioOutput_STFT,
    add_instrument,
    average_np_arrays
)


def get_triangle_mask(height, width):
    """
    Generate a triangle mask of shape (height, width) using a fixed slope.
    The mask is filled with 1s where the condition i < slope * j is met, and 0s elsewhere.

    Args:
        height (int): Height of the mask.
        width (int): Width of the mask.

    Returns:
        np.ndarray: A 2D numpy array representing the triangle mask.
    """
    mask = np.zeros((height, width))
    slope = 8 / 3  # Fixed slope value
    for i in range(height):
        for j in range(width):
            if i < slope * j:
                mask[i, j] = 1
    return mask


def get_inpaint_with_text_module(gradioWebUI, inpaintWithText_state, virtual_instruments_state):
    """
    Build the inpainting module with text guidance. This function sets up the inner functions
    for processing uploaded audio, sampling new sounds based on text prompts, showing sample outputs,
    and saving virtual instruments. It also builds the Gradio UI layout.

    Args:
        gradioWebUI: The main Gradio UI object containing model configurations.
        inpaintWithText_state: The Gradio state to store inpainting related data.
        virtual_instruments_state: The Gradio state to store virtual instrument information.
    """
    # Load configurations from the gradioWebUI object
    uNet = gradioWebUI.uNet
    freq_resolution, time_resolution = gradioWebUI.freq_resolution, gradioWebUI.time_resolution
    VAE_scale = gradioWebUI.VAE_scale
    height, width, channels = int(freq_resolution / VAE_scale), int(time_resolution / VAE_scale), gradioWebUI.channels
    timesteps = gradioWebUI.timesteps
    VAE_encoder = gradioWebUI.VAE_encoder
    VAE_quantizer = gradioWebUI.VAE_quantizer
    VAE_decoder = gradioWebUI.VAE_decoder
    CLAP = gradioWebUI.CLAP
    CLAP_tokenizer = gradioWebUI.CLAP_tokenizer
    device = gradioWebUI.device
    squared = gradioWebUI.squared
    sample_rate = gradioWebUI.sample_rate
    noise_strategy = gradioWebUI.noise_strategy

    def receive_upload_origin_audio(sound2sound_duration, sound2sound_origin, inpaintWithText_dict):
        """
        Process the uploaded or recorded source audio. This function normalizes the audio,
        adjusts its length, computes its STFT, and encodes it into latent representations.
        The latent representations and their Gradio image conversions are stored in inpaintWithText_dict.

        Args:
            sound2sound_duration (float): The duration for sound-to-sound conversion.
            sound2sound_origin (tuple): A tuple (origin_sr, origin_audio) of the sample rate and audio signal.
            inpaintWithText_dict (dict): A dictionary to store inpainting-related data.

        Returns:
            dict: A dictionary containing processed spectrogram images, phase images,
                  latent representation images, and updated inpaintWithText_state.
        """
        print(sound2sound_origin)
        origin_sr, origin_audio = sound2sound_origin

        # Normalize the audio signal
        origin_audio = origin_audio / np.max(np.abs(origin_audio))

        # Calculate the width based on time resolution, duration, and VAE scale
        width = int(time_resolution * ((sound2sound_duration + 1) / 4) / VAE_scale)
        # Calculate the expected audio length based on VAE scale and width
        audio_length = 256 * (VAE_scale * width - 1)
        print(f"audio_length: {audio_length}")
        # Adjust the length of the audio to match the expected audio_length
        origin_audio = adjust_audio_length(origin_audio, audio_length, origin_sr, sample_rate)

        # Compute the Short-Time Fourier Transform (STFT) of the audio
        D = librosa.stft(origin_audio, n_fft=1024, hop_length=256, win_length=1024)
        # Pad the STFT to match the required shape
        padded_D = pad_STFT(D)
        # Encode the STFT using a custom encoding function
        encoded_D = encode_stft(padded_D)

        # Convert the encoded STFT to a torch tensor and repeat along the batch dimension (batch size = 1)
        origin_spectrogram_batch_tensor = torch.from_numpy(
            np.repeat(encoded_D[np.newaxis, :, :, :], 1, axis=0)
        ).float().to(device)

        # Encode the spectrogram batch using the VAE encoder and quantizer to obtain latent representations
        origin_flipped_log_spectrums, origin_flipped_phases, origin_signals, origin_latent_representations, quantized_origin_latent_representations = InputBatch2Encode_STFT(
            VAE_encoder,
            origin_spectrogram_batch_tensor,
            resolution=(512, width * VAE_scale),
            quantizer=VAE_quantizer,
            squared=squared
        )

        # Store the latent representations in the inpaintWithText_dict (converted to list for JSON-compatibility)
        inpaintWithText_dict["origin_upload_latent_representations"] = origin_latent_representations.tolist()
        inpaintWithText_dict[
            "sound2sound_origin_upload_latent_representation_image"] = latent_representation_to_Gradio_image(
            origin_latent_representations[0]
        ).tolist()
        inpaintWithText_dict[
            "sound2sound_origin_upload_quantized_latent_representation_image"] = latent_representation_to_Gradio_image(
            quantized_origin_latent_representations[0]
        ).tolist()
        # Return processed images and update the state; some fields are updated via gr.update()
        return {
            sound2sound_origin_spectrogram_image: origin_flipped_log_spectrums[0],
            sound2sound_origin_phase_image: origin_flipped_phases[0],
            sound2sound_origin_upload_latent_representation_image: latent_representation_to_Gradio_image(
                origin_latent_representations[0]),
            sound2sound_origin_upload_quantized_latent_representation_image: latent_representation_to_Gradio_image(
                quantized_origin_latent_representations[0]),
            sound2sound_origin_microphone_latent_representation_image: gr.update(),
            sound2sound_origin_microphone_quantized_latent_representation_image: gr.update(),
            inpaintWithText_state: inpaintWithText_dict
        }

    def sound2sound_sample(
            sound2sound_origin_spectrogram,
            text2sound_prompts,
            text2sound_negative_prompts,
            sound2sound_batchsize,
            sound2sound_guidance_scale,
            sound2sound_sampler,
            sound2sound_sample_steps,
            sound2sound_noising_strength,
            sound2sound_seed,
            sound2sound_inpaint_area,
            mask_time_begin,
            mask_time_end,
            mask_frequency_begin,
            mask_frequency_end,
            inpaintWithText_dict
    ):
        """
        Generate new sound samples based on the uploaded origin spectrogram and text prompts.
        This function performs the following steps:
            1. Preprocesses inputs (seed, batch size, etc.).
            2. Encodes the text prompt to obtain a conditioning embedding.
            3. Processes the origin spectrogram to derive an average transparency mask.
            4. Retrieves the latent representations from the uploaded audio.
            5. Constructs a latent mask (with zoom and clipping) and applies user-specified mask regions.
            6. Depending on the chosen inpainting area ('masked' or 'unmasked'), inverts the mask.
            7. Runs the inpainting sampling process using DiffSynthSampler.
            8. Quantizes and decodes the new latent representations to obtain new spectrograms, phases, and audio signals.
            9. Aggregates the outputs into lists for batch processing.
            10. Updates inpaintWithText_dict with the new results.

        Args:
            sound2sound_origin_spectrogram (dict): The spectrogram of the uploaded audio.
            text2sound_prompts (str): The positive text prompt.
            text2sound_negative_prompts (str): The negative text prompt.
            sound2sound_batchsize (int): The batch size for sampling.
            sound2sound_guidance_scale (int): The guidance scale for classifier-free guidance.
            sound2sound_sampler: The selected sampler.
            sound2sound_sample_steps (int): Number of sampling steps.
            sound2sound_noising_strength (float): Noising strength parameter.
            sound2sound_seed (int/str): Seed value for random generation.
            sound2sound_inpaint_area (str): Choice between "masked" and "unmasked" inpainting area.
            mask_time_begin (float): Starting time for mask region.
            mask_time_end (float): Ending time for mask region.
            mask_frequency_begin (int): Starting frequency index for mask region.
            mask_frequency_end (int): Ending frequency index for mask region.
            inpaintWithText_dict (dict): Dictionary to store inpainting-related data.

        Returns:
            dict: Updated outputs including new latent representation images, spectrograms, phase images,
                  audio signal, updated sample index slider, seed textbox, and state dictionary.
        """
        # Preprocess input seed and batch size
        sound2sound_seed = safe_int(sound2sound_seed, 12345678)
        sound2sound_batchsize = int(sound2sound_batchsize)
        noising_strength = sound2sound_noising_strength
        sound2sound_sample_steps = int(sound2sound_sample_steps)
        CFG = int(sound2sound_guidance_scale)

        # Obtain text embedding for the positive prompt using the CLAP model
        text2sound_embedding = CLAP.get_text_features(
            **CLAP_tokenizer([text2sound_prompts], padding=True, return_tensors="pt")
        )[0].to(device)

        # Average the transparency layers from the origin spectrogram (provided by the image editor)
        averaged_transparency = average_np_arrays(sound2sound_origin_spectrogram["layers"])
        averaged_transparency = averaged_transparency[:, :, -1]  # Use the last channel

        # Retrieve the origin latent representations from the state dictionary and repeat for batch
        origin_latent_representations = torch.tensor(
            inpaintWithText_dict["origin_upload_latent_representations"]
        ).repeat(sound2sound_batchsize, 1, 1, 1).to(device)

        # Create a merged mask based on the averaged transparency (thresholded to 0 or 1)
        merged_mask = np.where(averaged_transparency > 0, 1, 0)
        print(f"np.shape(merged_mask): {np.shape(merged_mask)}")
        print(f"np.mean(merged_mask): {np.mean(merged_mask)}")
        # Adjust the resolution of the mask using scipy's zoom
        latent_mask = zoom(merged_mask, (1 / VAE_scale, 1 / VAE_scale))
        latent_mask = np.clip(latent_mask, 0, 1)
        # Set the user-specified masked region to 1 (based on time and frequency ranges)
        latent_mask[int(mask_frequency_begin):int(mask_frequency_end),
        int(mask_time_begin * time_resolution / (VAE_scale * 4)):
        int(mask_time_end * time_resolution / (VAE_scale * 4))] = 1

        # Invert the mask if the selected inpaint area is "masked"
        if sound2sound_inpaint_area == "masked":
            latent_mask = 1 - latent_mask
        # Convert the mask to a torch tensor and adjust dimensions: [batch, channels, height, width]
        latent_mask = torch.from_numpy(latent_mask).unsqueeze(0).unsqueeze(1).repeat(
            sound2sound_batchsize, channels, 1, 1
        ).float().to(device)
        # Flip the mask vertically (if needed)
        latent_mask = torch.flip(latent_mask, [2])

        # Initialize the DiffSynthSampler for inpainting with the given parameters
        mySampler = DiffSynthSampler(timesteps, height=height, channels=channels, noise_strategy=noise_strategy)
        # Get unconditional condition from negative prompts for classifier-free guidance
        unconditional_condition = CLAP.get_text_features(
            **CLAP_tokenizer([text2sound_negative_prompts], padding=True, return_tensors="pt")
        )[0]
        mySampler.activate_classifier_free_guidance(CFG, unconditional_condition.to(device))

        # Adjust the sample steps based on noising strength
        normalized_sample_steps = int(sound2sound_sample_steps / noising_strength)
        # Respace the timesteps for sampling
        mySampler.respace(list(np.linspace(0, timesteps - 1, normalized_sample_steps, dtype=np.int32)))

        # Get the width from the shape of the origin latent representations
        width = origin_latent_representations.shape[-1]
        # Repeat the text embedding to match the batch size
        condition = text2sound_embedding.repeat(sound2sound_batchsize, 1)

        # Perform the inpainting sampling to generate new latent representations
        new_sound_latent_representations, initial_noise = mySampler.inpaint_sample(
            model=uNet,
            shape=(sound2sound_batchsize, channels, height, width),
            seed=sound2sound_seed,
            noising_strength=noising_strength,
            guide_img=origin_latent_representations,
            mask=latent_mask,
            return_tensor=True,
            condition=condition,
            sampler=sound2sound_sampler
        )

        # Use the last latent representation from the sampling process
        new_sound_latent_representations = new_sound_latent_representations[-1]

        # Quantize the new sound latent representations using the VAE quantizer
        quantized_new_sound_latent_representations, loss, (_, _, _) = VAE_quantizer(new_sound_latent_representations)
        # Decode the quantized latent representations to obtain spectrograms, phases, and audio signals
        new_sound_flipped_log_spectrums, new_sound_flipped_phases, new_sound_signals, _, _, _ = encodeBatch2GradioOutput_STFT(
            VAE_decoder,
            quantized_new_sound_latent_representations,
            resolution=(512, width * VAE_scale),
            original_STFT_batch=None
        )

        # Initialize lists to collect outputs for each sample in the batch
        new_sound_latent_representation_gradio_images = []
        new_sound_quantized_latent_representation_gradio_images = []
        new_sound_spectrogram_gradio_images = []
        new_sound_phase_gradio_images = []
        new_sound_rec_signals_gradio = []

        # For each sample in the batch, convert latent representations to Gradio images and store the outputs
        for i in range(sound2sound_batchsize):
            new_sound_latent_representation_gradio_images.append(
                latent_representation_to_Gradio_image(new_sound_latent_representations[i])
            )
            new_sound_quantized_latent_representation_gradio_images.append(
                latent_representation_to_Gradio_image(quantized_new_sound_latent_representations[i])
            )
            new_sound_spectrogram_gradio_images.append(new_sound_flipped_log_spectrums[i])
            new_sound_phase_gradio_images.append(new_sound_flipped_phases[i])
            new_sound_rec_signals_gradio.append((sample_rate, new_sound_signals[i]))

        # Update the inpaintWithText_dict with all new generated images and latent representations
        inpaintWithText_dict[
            "new_sound_latent_representation_gradio_images"] = new_sound_latent_representation_gradio_images
        inpaintWithText_dict[
            "new_sound_quantized_latent_representation_gradio_images"] = new_sound_quantized_latent_representation_gradio_images
        inpaintWithText_dict["new_sound_spectrogram_gradio_images"] = new_sound_spectrogram_gradio_images
        inpaintWithText_dict["new_sound_phase_gradio_images"] = new_sound_phase_gradio_images
        inpaintWithText_dict["new_sound_rec_signals_gradio"] = new_sound_rec_signals_gradio

        # Store the raw latent representations in the state dictionary (converted to CPU numpy arrays)
        inpaintWithText_dict["latent_representations"] = new_sound_latent_representations.to("cpu").detach().numpy()
        inpaintWithText_dict["quantized_latent_representations"] = quantized_new_sound_latent_representations.to(
            "cpu").detach().numpy()
        inpaintWithText_dict["sampler"] = sound2sound_sampler

        # Return a dictionary with updated outputs to update the UI components
        return {
            sound2sound_new_sound_latent_representation_image: latent_representation_to_Gradio_image(
                new_sound_latent_representations[0]),
            sound2sound_new_sound_quantized_latent_representation_image: latent_representation_to_Gradio_image(
                quantized_new_sound_latent_representations[0]),
            sound2sound_new_sound_spectrogram_image: new_sound_flipped_log_spectrums[0],
            sound2sound_new_sound_phase_image: new_sound_flipped_phases[0],
            sound2sound_new_sound_audio: (sample_rate, new_sound_signals[0]),
            sound2sound_sample_index_slider: gr.update(
                minimum=0,
                maximum=sound2sound_batchsize - 1,
                value=0,
                step=1.0,
                visible=True,
                label="Sample index",
                info="Swipe to view other samples"
            ),
            sound2sound_seed_textbox: sound2sound_seed,
            inpaintWithText_state: inpaintWithText_dict
        }

    def show_sound2sound_sample(sound2sound_sample_index, inpaintWithText_dict):
        """
        Display the outputs for a specific sample index from the generated batch.

        Args:
            sound2sound_sample_index (int): The index of the sample to display.
            inpaintWithText_dict (dict): Dictionary containing lists of generated images and audio signals.

        Returns:
            dict: A dictionary with the images and audio corresponding to the selected sample index.
        """
        sample_index = int(sound2sound_sample_index)
        return {
            sound2sound_new_sound_latent_representation_image:
                inpaintWithText_dict["new_sound_latent_representation_gradio_images"][sample_index],
            sound2sound_new_sound_quantized_latent_representation_image:
                inpaintWithText_dict["new_sound_quantized_latent_representation_gradio_images"][sample_index],
            sound2sound_new_sound_spectrogram_image: inpaintWithText_dict["new_sound_spectrogram_gradio_images"][
                sample_index],
            sound2sound_new_sound_phase_image: inpaintWithText_dict["new_sound_phase_gradio_images"][sample_index],
            sound2sound_new_sound_audio: inpaintWithText_dict["new_sound_rec_signals_gradio"][sample_index]
        }

    def save_virtual_instrument(sample_index, virtual_instrument_name, sound2sound_dict, virtual_instruments_dict):
        """
        Save a virtual instrument using the generated sound sample.
        This function calls add_instrument to update the virtual_instruments_dict with the new instrument.

        Args:
            sample_index (int): Index of the sample to save.
            virtual_instrument_name (str): The name to assign to the virtual instrument.
            sound2sound_dict (dict): Dictionary containing sound2sound related data.
            virtual_instruments_dict (dict): Dictionary storing all virtual instruments.

        Returns:
            dict: Updated virtual_instruments_state and an updated textbox indicating the saved name.
        """
        virtual_instruments_dict = add_instrument(
            sound2sound_dict,
            virtual_instruments_dict,
            virtual_instrument_name,
            sample_index
        )
        return {
            virtual_instruments_state: virtual_instruments_dict,
            sound2sound_instrument_name_textbox: gr.Textbox(
                label="Instrument name",
                lines=1,
                placeholder=f"Saved as {virtual_instrument_name}!"
            )
        }

    # Build the Gradio UI components inside a Tab labeled "Inpaint"
    with gr.Tab("Inpaint"):
        # Display instructions for the user
        gr.Markdown("Upload a musical note and select the area by drawing on \"Input spectrogram\" for inpainting!")
        with gr.Row(variant="panel"):
            # Left column: Text prompt inputs
            with gr.Column(scale=3):
                text2sound_prompts_textbox = gr.Textbox(label="Positive prompt", lines=2, value="organ")
                text2sound_negative_prompts_textbox = gr.Textbox(label="Negative prompt", lines=2, value="")

            # Right column: Generate button and sample index slider (initially hidden)
            with gr.Column(scale=1):
                sound2sound_sample_button = gr.Button(variant="primary", value="Generate", scale=1)
                sound2sound_sample_index_slider = gr.Slider(
                    minimum=0, maximum=3, value=0, step=1.0, visible=False,
                    label="Sample index",
                    info="Swipe to view other samples"
                )

        with gr.Row(variant="panel"):
            # Left column: Audio and mask settings
            with gr.Column(scale=1):
                # Slider for selecting the duration of the sound-to-sound conversion
                sound2sound_duration_slider = gradioWebUI.get_duration_slider()
                # Audio component to either upload or record source sound
                sound2sound_origin_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    label="Upload/Record source sound",
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#01C6FF",
                        waveform_progress_color="#0066B4",
                        skip_length=1,
                        show_controls=False,
                    ),
                )
                # Grouping settings into tabs
                with gr.Row(variant="panel"):
                    with gr.Tab("Sound2sound settings"):
                        sound2sound_sample_steps_slider = gradioWebUI.get_sample_steps_slider()
                        sound2sound_sampler_radio = gradioWebUI.get_sampler_radio()
                        sound2sound_batchsize_slider = gradioWebUI.get_batchsize_slider()
                        sound2sound_noising_strength_slider = gradioWebUI.get_noising_strength_slider()
                        sound2sound_guidance_scale_slider = gradioWebUI.get_guidance_scale_slider()
                        sound2sound_seed_textbox = gradioWebUI.get_seed_textbox()
                    with gr.Tab("Mask prototypes"):
                        with gr.Tab("Mask along time axis"):
                            mask_time_begin_slider = gr.Slider(minimum=0.0, maximum=4.00, value=0.0, step=0.01,
                                                               label="Begin time")
                            mask_time_end_slider = gr.Slider(minimum=0.0, maximum=4.00, value=0.0, step=0.01,
                                                             label="End time")
                        with gr.Tab("Mask along frequency axis"):
                            mask_frequency_begin_slider = gr.Slider(minimum=0, maximum=127, value=0, step=1,
                                                                    label="Begin freq pixel")
                            mask_frequency_end_slider = gr.Slider(minimum=0, maximum=127, value=0, step=1,
                                                                  label="End freq pixel")
            # Right column: Display for spectrograms, images, and audio output
            with gr.Column(scale=1):
                with gr.Row(variant="panel"):
                    sound2sound_origin_spectrogram_image = gr.ImageEditor(
                        label="Input spectrogram (draw here!)",
                        type="numpy",
                        visible=True,
                        height=600,
                        scale=1
                    )
                    sound2sound_new_sound_spectrogram_image = gr.Image(
                        label="New sound spectrogram",
                        type="numpy",
                        height=600,
                        scale=1
                    )
                with gr.Row(variant="panel"):
                    sound2sound_inpaint_area_radio = gr.Radio(
                        label="Inpainting area", choices=["masked", "unmasked"],
                        value="masked", scale=1
                    )
                    sound2sound_new_sound_audio = gr.Audio(
                        type="numpy",
                        label="Play new sound",
                        interactive=False,
                        waveform_options=gr.WaveformOptions(
                            waveform_color="#FFB6C1",
                            waveform_progress_color="#FF0000",
                            skip_length=1,
                            show_controls=False,
                        ),
                        scale=1
                    )
                with gr.Row(variant="panel"):
                    sound2sound_instrument_name_textbox = gr.Textbox(
                        label="Instrument name",
                        lines=1,
                        placeholder="Name of your instrument"
                    )
                    sound2sound_save_instrument_button = gr.Button(
                        variant="primary",
                        value="Save instrument",
                        scale=1
                    )

        # Additional hidden components for displaying latent representations and phases
        with gr.Row(variant="panel"):
            sound2sound_origin_upload_latent_representation_image = gr.Image(
                label="Original latent representation",
                type="numpy",
                height=800,
                visible=False
            )
            sound2sound_origin_upload_quantized_latent_representation_image = gr.Image(
                label="Original quantized latent representation",
                type="numpy",
                height=800,
                visible=False
            )
            sound2sound_origin_microphone_latent_representation_image = gr.Image(
                label="Original latent representation",
                type="numpy",
                height=800,
                visible=False
            )
            sound2sound_origin_microphone_quantized_latent_representation_image = gr.Image(
                label="Original quantized latent representation",
                type="numpy",
                height=800,
                visible=False
            )
            sound2sound_new_sound_latent_representation_image = gr.Image(
                label="New latent representation",
                type="numpy",
                height=800,
                visible=False
            )
            sound2sound_new_sound_quantized_latent_representation_image = gr.Image(
                label="New sound quantized latent representation",
                type="numpy",
                height=800,
                visible=False
            )
            sound2sound_origin_phase_image = gr.Image(
                label="Original upload phase",
                type="numpy",
                visible=False
            )
            sound2sound_new_sound_phase_image = gr.Image(
                label="New sound phase",
                type="numpy",
                height=600,
                scale=1,
                visible=False
            )

    # Bind event: When the origin audio is changed (uploaded/recorded), process it
    sound2sound_origin_audio.change(
        receive_upload_origin_audio,
        inputs=[sound2sound_duration_slider, sound2sound_origin_audio, inpaintWithText_state],
        outputs=[
            sound2sound_origin_spectrogram_image,
            sound2sound_origin_phase_image,
            sound2sound_origin_upload_latent_representation_image,
            sound2sound_origin_upload_quantized_latent_representation_image,
            sound2sound_origin_microphone_latent_representation_image,
            sound2sound_origin_microphone_quantized_latent_representation_image,
            inpaintWithText_state
        ]
    )

    # Bind event: When the "Generate" button is clicked, generate new sound samples
    sound2sound_sample_button.click(
        sound2sound_sample,
        inputs=[
            sound2sound_origin_spectrogram_image,
            text2sound_prompts_textbox,
            text2sound_negative_prompts_textbox,
            sound2sound_batchsize_slider,
            sound2sound_guidance_scale_slider,
            sound2sound_sampler_radio,
            sound2sound_sample_steps_slider,
            sound2sound_noising_strength_slider,
            sound2sound_seed_textbox,
            sound2sound_inpaint_area_radio,
            mask_time_begin_slider,
            mask_time_end_slider,
            mask_frequency_begin_slider,
            mask_frequency_end_slider,
            inpaintWithText_state
        ],
        outputs=[
            sound2sound_new_sound_latent_representation_image,
            sound2sound_new_sound_quantized_latent_representation_image,
            sound2sound_new_sound_spectrogram_image,
            sound2sound_new_sound_phase_image,
            sound2sound_new_sound_audio,
            sound2sound_sample_index_slider,
            sound2sound_seed_textbox,
            inpaintWithText_state
        ]
    )

    # Bind event: When the sample index slider changes, show the corresponding sample output
    sound2sound_sample_index_slider.change(
        show_sound2sound_sample,
        inputs=[sound2sound_sample_index_slider, inpaintWithText_state],
        outputs=[
            sound2sound_new_sound_latent_representation_image,
            sound2sound_new_sound_quantized_latent_representation_image,
            sound2sound_new_sound_spectrogram_image,
            sound2sound_new_sound_phase_image,
            sound2sound_new_sound_audio
        ]
    )

    # Bind event: When the "Save instrument" button is clicked, save the instrument into the virtual instruments state
    sound2sound_save_instrument_button.click(
        save_virtual_instrument,
        inputs=[
            sound2sound_sample_index_slider,
            sound2sound_instrument_name_textbox,
            inpaintWithText_state,
            virtual_instruments_state
        ],
        outputs=[
            virtual_instruments_state,
            sound2sound_instrument_name_textbox
        ]
    )
