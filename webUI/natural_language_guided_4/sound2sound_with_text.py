import gradio as gr
import librosa
import numpy as np
import torch

from model.DiffSynthSampler import DiffSynthSampler
from tools import pad_STFT, encode_stft
from tools import safe_int, adjust_audio_length
from webUI.natural_language_guided_4.utils import InputBatch2Encode_STFT, encodeBatch2GradioOutput_STFT, \
    latent_representation_to_Gradio_image, resize_image_to_aspect_ratio, add_instrument


def get_sound2sound_with_text_module(gradioWebUI, sound2sound_with_text_state, virtual_instruments_state):
    """
    Build the Sound2Sound module with text guidance.

    This module processes an uploaded or recorded source sound and then allows the user to generate new
    sound variations based on text prompts (both positive and negative). It supports viewing and saving
    latent representations and synthesized spectrograms.

    Args:
        gradioWebUI: The main Gradio UI object containing model configurations.
        sound2sound_with_text_state: A Gradio state variable for storing sound-to-sound related data.
        virtual_instruments_state: A Gradio state variable for storing virtual instrument data.

    Returns:
        None. (This function builds and binds Gradio UI components.)
    """
    # Load configurations from the GradioWebUI object.
    uNet = gradioWebUI.uNet
    freq_resolution, time_resolution = gradioWebUI.freq_resolution, gradioWebUI.time_resolution
    VAE_scale = gradioWebUI.VAE_scale
    height = int(freq_resolution / VAE_scale)
    width = int(time_resolution / VAE_scale)
    channels = gradioWebUI.channels
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

    def receive_upload_origin_audio(sound2sound_duration, sound2sound_origin,
                                    sound2sound_with_text_dict, virtual_instruments_dict):
        """
        Process the uploaded/recorded origin audio for sound-to-sound transformation.

        This function:
          - Normalizes the audio.
          - Adjusts its length to the expected duration.
          - Computes its STFT, pads, and encodes it.
          - Computes latent representations via the VAE encoder and quantizer.
          - Stores the latent representations and their corresponding Gradio images in the state dictionary.

        Args:
            sound2sound_duration (float): The duration parameter for sound transformation.
            sound2sound_origin (tuple): A tuple (origin_sr, origin_audio) containing the sample rate and audio signal.
            sound2sound_with_text_dict (dict): A dictionary to store sound2sound-related data.
            virtual_instruments_dict (dict): The current state of virtual instruments.

        Returns:
            dict: Updated outputs for the UI components:
                - The processed spectrogram image.
                - The processed phase image.
                - The latent representation image.
                - The quantized latent representation image.
                - Updated state dictionaries.
        """
        origin_sr, origin_audio = sound2sound_origin
        # Normalize the audio signal.
        origin_audio = origin_audio / np.max(np.abs(origin_audio))

        # Calculate the width based on the duration and VAE scale.
        width = int(time_resolution * ((sound2sound_duration + 1) / 4) / VAE_scale)
        # Compute expected audio length.
        audio_length = 256 * (VAE_scale * width - 1)
        # Adjust the audio length.
        origin_audio = adjust_audio_length(origin_audio, audio_length, origin_sr, sample_rate)

        # Compute the Short-Time Fourier Transform (STFT) of the audio.
        D = librosa.stft(origin_audio, n_fft=1024, hop_length=256, win_length=1024)
        # Pad the STFT to match expected dimensions.
        padded_D = pad_STFT(D)
        # Encode the padded STFT.
        encoded_D = encode_stft(padded_D)

        # Convert the encoded STFT into a torch tensor and repeat to simulate a batch of size 1.
        origin_spectrogram_batch_tensor = torch.from_numpy(
            np.repeat(encoded_D[np.newaxis, :, :, :], 1, axis=0)
        ).float().to(device)

        # Encode the spectrogram batch using the VAE encoder and quantizer.
        origin_flipped_log_spectrums, origin_flipped_phases, origin_signals, \
            origin_latent_representations, quantized_origin_latent_representations = InputBatch2Encode_STFT(
            VAE_encoder,
            origin_spectrogram_batch_tensor,
            resolution=(512, width * VAE_scale),
            quantizer=VAE_quantizer,
            squared=squared
        )

        # Store latent representations (converted to list for JSON compatibility) in the state dictionary.
        sound2sound_with_text_dict["origin_latent_representations"] = origin_latent_representations.tolist()
        sound2sound_with_text_dict["sound2sound_origin_latent_representation_image"] = \
            latent_representation_to_Gradio_image(origin_latent_representations[0]).tolist()
        sound2sound_with_text_dict["sound2sound_origin_quantized_latent_representation_image"] = \
            latent_representation_to_Gradio_image(quantized_origin_latent_representations[0]).tolist()

        # Return updates for the UI components.
        return {
            sound2sound_origin_spectrogram_image: resize_image_to_aspect_ratio(origin_flipped_log_spectrums[0], 1.55,
                                                                               1),
            sound2sound_origin_phase_image: resize_image_to_aspect_ratio(origin_flipped_phases[0], 1.55, 1),
            sound2sound_origin_latent_representation_image: latent_representation_to_Gradio_image(
                origin_latent_representations[0]),
            sound2sound_origin_quantized_latent_representation_image: latent_representation_to_Gradio_image(
                quantized_origin_latent_representations[0]),
            sound2sound_with_text_state: sound2sound_with_text_dict,
            virtual_instruments_state: virtual_instruments_dict
        }

    def sound2sound_sample(sound2sound_prompts, sound2sound_negative_prompts, sound2sound_batchsize,
                           sound2sound_guidance_scale, sound2sound_sampler,
                           sound2sound_sample_steps, sound2sound_noising_strength, sound2sound_seed,
                           sound2sound_dict, virtual_instruments_dict):
        """
        Generate new sound samples based on text prompts and the origin latent representation.

        Steps include:
          - Processing the input seed, batch size, and sampling parameters.
          - Repeating the origin latent representation for the batch.
          - Obtaining text embeddings for the positive and negative prompts.
          - Initializing the DiffSynthSampler for image-guided sampling.
          - Activating classifier-free guidance with the negative prompt.
          - Adjusting sampling steps based on noising strength.
          - Generating new latent representations.
          - Quantizing and decoding the new latent representations into spectrograms, phase images, and audio.
          - Storing the generated outputs in the state dictionary.

        Args:
            sound2sound_prompts (str): The positive text prompt.
            sound2sound_negative_prompts (str): The negative text prompt.
            sound2sound_batchsize (int/str): The batch size for generation.
            sound2sound_guidance_scale (int/str): The guidance scale for text conditioning.
            sound2sound_sampler: The selected sampler.
            sound2sound_sample_steps (int/str): The number of sampling steps.
            sound2sound_noising_strength (float): The noising strength parameter.
            sound2sound_seed (int/str): The random seed.
            sound2sound_dict (dict): The state dictionary containing origin latent representations.
            virtual_instruments_dict (dict): The current virtual instruments state.

        Returns:
            dict: Updated UI outputs with new latent images, spectrograms, phase images, and audio.
        """
        # Process input parameters.
        sound2sound_seed = safe_int(sound2sound_seed, 12345678)
        sound2sound_batchsize = int(sound2sound_batchsize)
        noising_strength = sound2sound_noising_strength
        sound2sound_sample_steps = int(sound2sound_sample_steps)
        CFG = int(sound2sound_guidance_scale)

        # Retrieve and repeat the origin latent representations for the given batch size.
        origin_latent_representations = torch.tensor(sound2sound_dict["origin_latent_representations"]) \
            .repeat(sound2sound_batchsize, 1, 1, 1).to(device)

        # Obtain text embedding for the positive prompt.
        text2sound_embedding = CLAP.get_text_features(
            **CLAP_tokenizer([sound2sound_prompts], padding=True, return_tensors="pt")
        )[0].to(device)

        # Initialize the DiffSynthSampler for inpainting.
        mySampler = DiffSynthSampler(timesteps, height=height, channels=channels, noise_strategy=noise_strategy)
        # Obtain text embedding for the negative prompt.
        negative_condition = CLAP.get_text_features(
            **CLAP_tokenizer([sound2sound_negative_prompts], padding=True, return_tensors="pt")
        )[0]
        # Activate classifier-free guidance.
        mySampler.activate_classifier_free_guidance(CFG, negative_condition.to(device))

        # Normalize the sample steps based on the noising strength.
        normalized_sample_steps = int(sound2sound_sample_steps / noising_strength)
        mySampler.respace(list(np.linspace(0, timesteps - 1, normalized_sample_steps, dtype=np.int32)))

        # Repeat the positive prompt embedding to match the batch size.
        condition = text2sound_embedding.repeat(sound2sound_batchsize, 1)

        # Determine the width from the origin latent representation.
        width = origin_latent_representations.shape[-1]
        # Generate new latent representations using image-guided sampling.
        new_sound_latent_representations, initial_noise = mySampler.img_guided_sample(
            model=uNet,
            shape=(sound2sound_batchsize, channels, height, width),
            seed=sound2sound_seed,
            noising_strength=noising_strength,
            guide_img=origin_latent_representations,
            return_tensor=True,
            condition=condition,
            sampler=sound2sound_sampler
        )
        # Select the final latent representation from the sampling process.
        new_sound_latent_representations = new_sound_latent_representations[-1]

        # Quantize the new latent representations.
        quantized_new_sound_latent_representations, loss, (_, _, _) = VAE_quantizer(new_sound_latent_representations)

        # Decode the quantized latent representations to obtain spectrograms, phase images, and audio signals.
        new_sound_flipped_log_spectrums, new_sound_flipped_phases, new_sound_signals, _, _, _ = encodeBatch2GradioOutput_STFT(
            VAE_decoder,
            quantized_new_sound_latent_representations,
            resolution=(512, width * VAE_scale),
            original_STFT_batch=None
        )

        # Initialize lists to collect generated outputs for each sample in the batch.
        new_sound_latent_representation_gradio_images = []
        new_sound_quantized_latent_representation_gradio_images = []
        new_sound_spectrogram_gradio_images = []
        new_sound_phase_gradio_images = []
        new_sound_rec_signals_gradio = []
        # Process each sample in the batch.
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
        # Store generated outputs and additional info in the state dictionary.
        sound2sound_dict[
            "new_sound_latent_representation_gradio_images"] = new_sound_latent_representation_gradio_images
        sound2sound_dict[
            "new_sound_quantized_latent_representation_gradio_images"] = new_sound_quantized_latent_representation_gradio_images
        sound2sound_dict["new_sound_spectrogram_gradio_images"] = new_sound_spectrogram_gradio_images
        sound2sound_dict["new_sound_phase_gradio_images"] = new_sound_phase_gradio_images
        sound2sound_dict["new_sound_rec_signals_gradio"] = new_sound_rec_signals_gradio

        # Save additional parameters for possible instrument saving.
        sound2sound_dict["latent_representations"] = new_sound_latent_representations.to("cpu").detach().numpy()
        sound2sound_dict["quantized_latent_representations"] = quantized_new_sound_latent_representations.to(
            "cpu").detach().numpy()
        sound2sound_dict["condition"] = condition.to("cpu").detach().numpy()
        sound2sound_dict["negative_condition"] = negative_condition.to("cpu").detach().numpy()
        sound2sound_dict["guidance_scale"] = CFG
        sound2sound_dict["sampler"] = sound2sound_sampler

        # Return a dictionary updating UI components with the generated outputs.
        return {
            sound2sound_new_sound_latent_representation_image: latent_representation_to_Gradio_image(
                new_sound_latent_representations[0]),
            sound2sound_new_sound_quantized_latent_representation_image: latent_representation_to_Gradio_image(
                quantized_new_sound_latent_representations[0]),
            sound2sound_new_sound_spectrogram_image: resize_image_to_aspect_ratio(new_sound_flipped_log_spectrums[0],
                                                                                  1.55, 1),
            sound2sound_new_sound_phase_image: resize_image_to_aspect_ratio(new_sound_flipped_phases[0], 1.55, 1),
            sound2sound_new_sound_audio: (sample_rate, new_sound_signals[0]),
            sound2sound_sample_index_slider: gr.update(minimum=0, maximum=sound2sound_batchsize - 1, value=0, step=1.0,
                                                       visible=True, label="Sample index",
                                                       info="Swipe to view other samples"),
            sound2sound_seed_textbox: sound2sound_seed,
            sound2sound_with_text_state: sound2sound_dict,
            virtual_instruments_state: virtual_instruments_dict
        }

    def show_sound2sound_sample(sound2sound_sample_index, sound2sound_with_text_dict):
        """
        Display outputs for a selected sample index from the generated batch.

        Args:
            sound2sound_sample_index (int): The index of the sample to display.
            sound2sound_with_text_dict (dict): The state dictionary containing generated outputs.

        Returns:
            dict: A dictionary updating UI components with outputs for the selected sample.
        """
        sample_index = int(sound2sound_sample_index)
        return {
            sound2sound_new_sound_latent_representation_image:
                sound2sound_with_text_dict["new_sound_latent_representation_gradio_images"][sample_index],
            sound2sound_new_sound_quantized_latent_representation_image:
                sound2sound_with_text_dict["new_sound_quantized_latent_representation_gradio_images"][sample_index],
            sound2sound_new_sound_spectrogram_image: resize_image_to_aspect_ratio(
                sound2sound_with_text_dict["new_sound_spectrogram_gradio_images"][sample_index], 1.55, 1),
            sound2sound_new_sound_phase_image: resize_image_to_aspect_ratio(
                sound2sound_with_text_dict["new_sound_phase_gradio_images"][sample_index], 1.55, 1),
            sound2sound_new_sound_audio: sound2sound_with_text_dict["new_sound_rec_signals_gradio"][sample_index]
        }

    def save_virtual_instrument(sample_index, virtual_instrument_name, sound2sound_dict, virtual_instruments_dict):
        """
        Save a generated sound sample as a virtual instrument.

        Args:
            sample_index (int): The index of the sample to save.
            virtual_instrument_name (str): The name for the virtual instrument.
            sound2sound_dict (dict): The state dictionary containing generated outputs.
            virtual_instruments_dict (dict): The current virtual instruments state.

        Returns:
            dict: Updated virtual_instruments_state and a textbox displaying the saved instrument name.
        """
        virtual_instruments_dict = add_instrument(sound2sound_dict, virtual_instruments_dict,
                                                  virtual_instrument_name, sample_index)
        return {
            virtual_instruments_state: virtual_instruments_dict,
            text2sound_instrument_name_textbox: gr.Textbox(label="Instrument name", lines=1,
                                                           placeholder=f"Saved as {virtual_instrument_name}!")
        }

    # Build the Gradio UI layout within a Tab titled "Sound2Sound".
    with gr.Tab("Sound2Sound"):
        gr.Markdown("Generate new sound based on a given sound!")
        with gr.Row(variant="panel"):
            # Column for text prompts.
            with gr.Column(scale=3):
                sound2sound_prompts_textbox = gr.Textbox(label="Positive prompt", lines=2, value="organ")
                text2sound_negative_prompts_textbox = gr.Textbox(label="Negative prompt", lines=2, value="")
            # Column for the Generate button and sample index slider.
            with gr.Column(scale=1):
                sound2sound_sample_button = gr.Button(variant="primary", value="Generate", scale=1)
                sound2sound_sample_index_slider = gr.Slider(minimum=0, maximum=3, value=0, step=1.0, visible=False,
                                                            label="Sample index", info="Swipe to view other samples")
        with gr.Row(variant="panel"):
            # Column for origin audio and settings.
            with gr.Column(scale=1):
                with gr.Tab("Origin sound"):
                    # Duration slider for the origin sound.
                    sound2sound_duration_slider = gradioWebUI.get_duration_slider()
                    # Audio input component for uploading/recording the source sound.
                    sound2sound_origin_audio = gr.Audio(
                        sources=["microphone", "upload"],
                        label="Upload/Record source sound",
                        waveform_options=gr.WaveformOptions(
                            waveform_color="#01C6FF",
                            waveform_progress_color="#0066B4",
                            skip_length=1,
                            show_controls=False,
                        )
                    )
                    with gr.Row(variant="panel"):
                        # Display components for the original spectrogram and phase images.
                        sound2sound_origin_spectrogram_image = gr.Image(label="Original upload spectrogram",
                                                                        type="numpy", visible=True)
                        sound2sound_origin_phase_image = gr.Image(label="Original upload phase",
                                                                  type="numpy", visible=True)
                with gr.Tab("Sound2sound settings"):
                    # Settings for sample steps, sampler selection, batch size, noising strength, guidance scale, and seed.
                    sound2sound_sample_steps_slider = gradioWebUI.get_sample_steps_slider()
                    sound2sound_sampler_radio = gradioWebUI.get_sampler_radio()
                    sound2sound_batchsize_slider = gradioWebUI.get_batchsize_slider()
                    sound2sound_noising_strength_slider = gradioWebUI.get_noising_strength_slider()
                    sound2sound_guidance_scale_slider = gradioWebUI.get_guidance_scale_slider()
                    sound2sound_seed_textbox = gradioWebUI.get_seed_textbox()
            # Column for the generated sound output.
            with gr.Column(scale=1):
                sound2sound_new_sound_audio = gr.Audio(type="numpy", label="Play new sound", interactive=False,
                                                       waveform_options=gr.WaveformOptions(
                                                           waveform_color="#FFB6C1",
                                                           waveform_progress_color="#FF0000",
                                                           skip_length=1,
                                                           show_controls=False
                                                       ))
                with gr.Row(variant="panel"):
                    # Components to display the new sound spectrogram and phase images.
                    sound2sound_new_sound_spectrogram_image = gr.Image(label="New sound spectrogram", type="numpy",
                                                                       scale=1)
                    sound2sound_new_sound_phase_image = gr.Image(label="New sound phase", type="numpy", scale=1)
                with gr.Row(variant="panel"):
                    # Components for saving the generated instrument.
                    text2sound_instrument_name_textbox = gr.Textbox(label="Instrument name", lines=2,
                                                                    placeholder="Name of your instrument", scale=1)
                    text2sound_save_instrument_button = gr.Button(variant="primary", value="Save instrument", scale=1)
        with gr.Row(variant="panel"):
            # Hidden components for displaying latent representations.
            sound2sound_origin_latent_representation_image = gr.Image(label="Original latent representation",
                                                                      type="numpy", height=800, visible=False)
            sound2sound_origin_quantized_latent_representation_image = gr.Image(
                label="Original quantized latent representation",
                type="numpy", height=800, visible=False)
            sound2sound_new_sound_latent_representation_image = gr.Image(label="New latent representation",
                                                                         type="numpy", height=800, visible=False)
            sound2sound_new_sound_quantized_latent_representation_image = gr.Image(
                label="New sound quantized latent representation",
                type="numpy", height=800, visible=False)

    # Bind the event for when the origin audio is changed (uploaded or recorded).
    sound2sound_origin_audio.change(
        receive_upload_origin_audio,
        inputs=[sound2sound_duration_slider, sound2sound_origin_audio, sound2sound_with_text_state,
                virtual_instruments_state],
        outputs=[sound2sound_origin_spectrogram_image, sound2sound_origin_phase_image,
                 sound2sound_origin_latent_representation_image,
                 sound2sound_origin_quantized_latent_representation_image,
                 sound2sound_with_text_state, virtual_instruments_state]
    )

    # Bind the event for when the "Generate" button is clicked.
    sound2sound_sample_button.click(
        sound2sound_sample,
        inputs=[sound2sound_prompts_textbox, text2sound_negative_prompts_textbox, sound2sound_batchsize_slider,
                sound2sound_guidance_scale_slider, sound2sound_sampler_radio, sound2sound_sample_steps_slider,
                sound2sound_noising_strength_slider, sound2sound_seed_textbox, sound2sound_with_text_state,
                virtual_instruments_state],
        outputs=[sound2sound_new_sound_latent_representation_image,
                 sound2sound_new_sound_quantized_latent_representation_image,
                 sound2sound_new_sound_spectrogram_image, sound2sound_new_sound_phase_image,
                 sound2sound_new_sound_audio,
                 sound2sound_sample_index_slider, sound2sound_seed_textbox, sound2sound_with_text_state,
                 virtual_instruments_state]
    )

    # Bind the event for saving a generated instrument.
    text2sound_save_instrument_button.click(
        save_virtual_instrument,
        inputs=[sound2sound_sample_index_slider, text2sound_instrument_name_textbox, sound2sound_with_text_state,
                virtual_instruments_state],
        outputs=[virtual_instruments_state, text2sound_instrument_name_textbox]
    )

    # Bind the event for changing the sample index slider to update displayed sample.
    sound2sound_sample_index_slider.change(
        show_sound2sound_sample,
        inputs=[sound2sound_sample_index_slider, sound2sound_with_text_state],
        outputs=[sound2sound_new_sound_latent_representation_image,
                 sound2sound_new_sound_quantized_latent_representation_image,
                 sound2sound_new_sound_spectrogram_image, sound2sound_new_sound_phase_image,
                 sound2sound_new_sound_audio]
    )
