import gradio as gr
import numpy as np

from model.DiffSynthSampler import DiffSynthSampler
from tools import safe_int
from webUI.natural_language_guided_4.utils import latent_representation_to_Gradio_image, \
    encodeBatch2GradioOutput_STFT, add_instrument, resize_image_to_aspect_ratio


def get_text2sound_module(gradioWebUI, text2sound_state, virtual_instruments_state):
    """
    Build the Text2Sound module that generates new sounds from text prompts.

    This module uses text prompts (both positive and negative) to condition a diffusion
    process that generates latent representations. These representations are then
    decoded into spectrograms, phase images, and audio. The module also supports saving
    the generated sound as a virtual instrument.

    Args:
        gradioWebUI: The main Gradio UI object containing model configurations.
        text2sound_state: A Gradio state variable to store text-to-sound related data.
        virtual_instruments_state: A Gradio state variable to store virtual instrument data.

    Returns:
        None. (This function builds and binds Gradio UI components for the text-to-sound workflow.)
    """
    # Load configurations from gradioWebUI
    uNet = gradioWebUI.uNet
    freq_resolution, time_resolution = gradioWebUI.freq_resolution, gradioWebUI.time_resolution
    VAE_scale = gradioWebUI.VAE_scale
    height = int(freq_resolution / VAE_scale)
    width = int(time_resolution / VAE_scale)
    channels = gradioWebUI.channels

    timesteps = gradioWebUI.timesteps
    VAE_quantizer = gradioWebUI.VAE_quantizer
    VAE_decoder = gradioWebUI.VAE_decoder
    CLAP = gradioWebUI.CLAP
    CLAP_tokenizer = gradioWebUI.CLAP_tokenizer
    device = gradioWebUI.device
    squared = gradioWebUI.squared
    sample_rate = gradioWebUI.sample_rate
    noise_strategy = gradioWebUI.noise_strategy

    def diffusion_random_sample(text2sound_prompts, text2sound_negative_prompts, text2sound_batchsize,
                                text2sound_duration, text2sound_guidance_scale, text2sound_sampler,
                                text2sound_sample_steps, text2sound_seed, text2sound_dict):
        """
        Generate a batch of sound samples using a diffusion process conditioned on text prompts.

        Steps:
          1. Convert input parameters (batch size, sample steps, seed) to the proper types.
          2. Calculate the output image width based on the provided duration and VAE scale.
          3. Extract the text embedding from the positive prompt.
          4. Initialize the DiffSynthSampler and activate classifier-free guidance using the negative prompt.
          5. Respace the timesteps and sample to generate new latent representations.
          6. Quantize and decode these latent representations to obtain spectrograms, phase images, and audio.
          7. Store all generated outputs in the state dictionary.

        Args:
            text2sound_prompts (str): The positive text prompt.
            text2sound_negative_prompts (str): The negative text prompt.
            text2sound_batchsize (int/str): Number of samples to generate.
            text2sound_duration (float): Duration parameter affecting the output width.
            text2sound_guidance_scale (int/str): Guidance scale for conditioning.
            text2sound_sampler: The chosen sampler type.
            text2sound_sample_steps (int/str): Number of sampling steps.
            text2sound_seed (int/str): Seed for random generation.
            text2sound_dict (dict): State dictionary to store generated outputs.

        Returns:
            dict: A dictionary of UI outputs including:
                  - The first latent representation image.
                  - The first quantized latent representation image.
                  - A sampled spectrogram, phase image, and audio.
                  - Updated seed textbox and state.
                  - A sample index slider for browsing the batch.
        """
        # Convert sample steps and seed to proper types.
        text2sound_sample_steps = int(text2sound_sample_steps)
        text2sound_seed = safe_int(text2sound_seed, 12345678)

        # Calculate the output width based on the given duration and VAE scale.
        width = int(time_resolution * ((text2sound_duration + 1) / 4) / VAE_scale)

        text2sound_batchsize = int(text2sound_batchsize)

        # Get the text embedding for the positive prompt.
        text2sound_embedding = CLAP.get_text_features(
            **CLAP_tokenizer([text2sound_prompts], padding=True, return_tensors="pt")
        )[0].to(device)

        CFG = int(text2sound_guidance_scale)

        # Initialize the DiffSynthSampler for the diffusion process.
        mySampler = DiffSynthSampler(timesteps, height=height, channels=channels, noise_strategy=noise_strategy)
        # Get the negative prompt embedding.
        negative_condition = CLAP.get_text_features(
            **CLAP_tokenizer([text2sound_negative_prompts], padding=True, return_tensors="pt")
        )[0]

        # Activate classifier-free guidance using the negative condition.
        mySampler.activate_classifier_free_guidance(CFG, negative_condition.to(device))

        # Set up the sampling timesteps.
        mySampler.respace(list(np.linspace(0, timesteps - 1, text2sound_sample_steps, dtype=np.int32)))

        # Repeat the text embedding to match the batch size.
        condition = text2sound_embedding.repeat(text2sound_batchsize, 1)

        # Generate latent representations using the sampler.
        latent_representations, initial_noise = mySampler.sample(
            model=uNet, shape=(text2sound_batchsize, channels, height, width),
            seed=text2sound_seed, return_tensor=True, condition=condition,
            sampler=text2sound_sampler
        )
        # Select the final latent representation from the sequence.
        latent_representations = latent_representations[-1]

        # Initialize lists to store UI images and audio for each sample.
        latent_representation_gradio_images = []
        quantized_latent_representation_gradio_images = []
        new_sound_spectrogram_gradio_images = []
        new_sound_phase_gradio_images = []
        new_sound_rec_signals_gradio = []

        # Quantize the latent representations.
        quantized_latent_representations, loss, (_, _, _) = VAE_quantizer(latent_representations)
        # Decode the quantized representations into spectrograms, phase images, and audio.
        flipped_log_spectrums, flipped_phases, rec_signals, _, _, _ = encodeBatch2GradioOutput_STFT(
            VAE_decoder, quantized_latent_representations,
            resolution=(512, width * VAE_scale),
            original_STFT_batch=None
        )

        # Process each sample in the batch.
        for i in range(text2sound_batchsize):
            latent_representation_gradio_images.append(
                latent_representation_to_Gradio_image(latent_representations[i])
            )
            quantized_latent_representation_gradio_images.append(
                latent_representation_to_Gradio_image(quantized_latent_representations[i])
            )
            new_sound_spectrogram_gradio_images.append(flipped_log_spectrums[i])
            new_sound_phase_gradio_images.append(flipped_phases[i])
            new_sound_rec_signals_gradio.append((sample_rate, rec_signals[i]))

        # Update the state dictionary with all generated outputs.
        text2sound_dict["latent_representation_gradio_images"] = latent_representation_gradio_images
        text2sound_dict["quantized_latent_representation_gradio_images"] = quantized_latent_representation_gradio_images
        text2sound_dict["new_sound_spectrogram_gradio_images"] = new_sound_spectrogram_gradio_images
        text2sound_dict["new_sound_phase_gradio_images"] = new_sound_phase_gradio_images
        text2sound_dict["new_sound_rec_signals_gradio"] = new_sound_rec_signals_gradio

        # Save additional parameters for potential instrument saving.
        text2sound_dict["latent_representations"] = latent_representations.to("cpu").detach().numpy()
        text2sound_dict["quantized_latent_representations"] = quantized_latent_representations.to(
            "cpu").detach().numpy()
        text2sound_dict["condition"] = condition.to("cpu").detach().numpy()
        text2sound_dict["negative_condition"] = negative_condition.to("cpu").detach().numpy()
        text2sound_dict["guidance_scale"] = CFG
        text2sound_dict["sampler"] = text2sound_sampler

        # Return UI outputs for the first generated sample and updated state.
        return {
            text2sound_latent_representation_image: text2sound_dict["latent_representation_gradio_images"][0],
            text2sound_quantized_latent_representation_image:
                text2sound_dict["quantized_latent_representation_gradio_images"][0],
            text2sound_sampled_spectrogram_image: resize_image_to_aspect_ratio(
                text2sound_dict["new_sound_spectrogram_gradio_images"][0], 1.55, 1),
            text2sound_sampled_phase_image: resize_image_to_aspect_ratio(
                text2sound_dict["new_sound_phase_gradio_images"][0], 1.55, 1),
            text2sound_sampled_audio: text2sound_dict["new_sound_rec_signals_gradio"][0],
            text2sound_seed_textbox: text2sound_seed,
            text2sound_state: text2sound_dict,
            text2sound_sample_index_slider: gr.update(minimum=0, maximum=text2sound_batchsize - 1, value=0, step=1,
                                                      visible=True, label="Sample index.",
                                                      info="Swipe to view other samples")
        }

    def show_random_sample(sample_index, text2sound_dict):
        """
        Update the UI to display outputs corresponding to a selected sample index.

        Args:
            sample_index (int): The index of the sample to display.
            text2sound_dict (dict): The state dictionary containing generated outputs.

        Returns:
            dict: UI outputs updated to reflect the selected sample.
        """
        sample_index = int(sample_index)
        text2sound_dict["sample_index"] = sample_index
        print(text2sound_dict["new_sound_rec_signals_gradio"][sample_index])
        return {
            text2sound_latent_representation_image: text2sound_dict["latent_representation_gradio_images"][
                sample_index],
            text2sound_quantized_latent_representation_image:
                text2sound_dict["quantized_latent_representation_gradio_images"][sample_index],
            text2sound_sampled_spectrogram_image: resize_image_to_aspect_ratio(
                text2sound_dict["new_sound_spectrogram_gradio_images"][sample_index], 1.55, 1),
            text2sound_sampled_phase_image: resize_image_to_aspect_ratio(
                text2sound_dict["new_sound_phase_gradio_images"][sample_index], 1.55, 1),
            text2sound_sampled_audio: text2sound_dict["new_sound_rec_signals_gradio"][sample_index]
        }

    def save_virtual_instrument(sample_index, virtual_instrument_name, text2sound_dict, virtual_instruments_dict):
        """
        Save a generated sound sample as a virtual instrument.

        Args:
            sample_index (int): The index of the sample to save.
            virtual_instrument_name (str): The name for the new virtual instrument.
            text2sound_dict (dict): State dictionary containing generated outputs.
            virtual_instruments_dict (dict): Current virtual instruments state.

        Returns:
            dict: Updated virtual instruments state and a textbox indicating the saved instrument name.
        """
        virtual_instruments_dict = add_instrument(text2sound_dict, virtual_instruments_dict,
                                                  virtual_instrument_name, sample_index)

        return {
            virtual_instruments_state: virtual_instruments_dict,
            text2sound_instrument_name_textbox: gr.Textbox(label="Instrument name", lines=1,
                                                           placeholder=f"Saved as {virtual_instrument_name}!")
        }

    # Build the Gradio UI layout within a Tab titled "Text2sound".
    with gr.Tab("Text2sound"):
        gr.Markdown("Use neural networks to select random sounds using your favorite instrument!")
        with gr.Row(variant="panel"):
            # Column for text prompt input.
            with gr.Column(scale=3):
                text2sound_prompts_textbox = gr.Textbox(label="Positive prompt", lines=2, value="string")
                text2sound_negative_prompts_textbox = gr.Textbox(label="Negative prompt", lines=2, value="")
            # Column for the sampling button and sample index slider.
            with gr.Column(scale=1):
                text2sound_sampling_button = gr.Button(
                    variant="primary",
                    value="Generate a batch of samples and show the first one",
                    scale=1
                )
                text2sound_sample_index_slider = gr.Slider(
                    minimum=0, maximum=3, value=0, step=1.0, visible=False,
                    label="Sample index",
                    info="Swipe to view other samples"
                )
        with gr.Row(variant="panel"):
            # Column for sampling settings.
            with gr.Column(variant="panel", scale=1):
                text2sound_sample_steps_slider = gradioWebUI.get_sample_steps_slider()
                text2sound_sampler_radio = gradioWebUI.get_sampler_radio()
                text2sound_batchsize_slider = gradioWebUI.get_batchsize_slider()
                text2sound_duration_slider = gradioWebUI.get_duration_slider()
                text2sound_guidance_scale_slider = gradioWebUI.get_guidance_scale_slider()
                text2sound_seed_textbox = gradioWebUI.get_seed_textbox()
            # Column for displaying generated outputs.
            with gr.Column(variant="panel", scale=1):
                with gr.Row(variant="panel"):
                    text2sound_sampled_spectrogram_image = gr.Image(label="Sampled spectrogram", type="numpy")
                    text2sound_sampled_phase_image = gr.Image(label="Sampled phase", type="numpy")
                text2sound_sampled_audio = gr.Audio(type="numpy", label="Play", scale=1)
                with gr.Row(variant="panel"):
                    text2sound_instrument_name_textbox = gr.Textbox(
                        label="Instrument name", lines=2,
                        placeholder="Name of your instrument", scale=1
                    )
                    text2sound_save_instrument_button = gr.Button(
                        variant="primary", value="Save instrument", scale=1
                    )
        with gr.Row(variant="panel"):
            # Hidden components for displaying latent representations.
            text2sound_latent_representation_image = gr.Image(
                label="Sampled latent representation", type="numpy",
                height=200, width=100, visible=False
            )
            text2sound_quantized_latent_representation_image = gr.Image(
                label="Quantized latent representation", type="numpy",
                height=200, width=100, visible=False
            )

    # Bind the event for generating samples.
    text2sound_sampling_button.click(
        diffusion_random_sample,
        inputs=[
            text2sound_prompts_textbox,
            text2sound_negative_prompts_textbox,
            text2sound_batchsize_slider,
            text2sound_duration_slider,
            text2sound_guidance_scale_slider,
            text2sound_sampler_radio,
            text2sound_sample_steps_slider,
            text2sound_seed_textbox,
            text2sound_state
        ],
        outputs=[
            text2sound_latent_representation_image,
            text2sound_quantized_latent_representation_image,
            text2sound_sampled_spectrogram_image,
            text2sound_sampled_phase_image,
            text2sound_sampled_audio,
            text2sound_seed_textbox,
            text2sound_state,
            text2sound_sample_index_slider
        ]
    )

    # Bind the event for saving the generated instrument.
    text2sound_save_instrument_button.click(
        save_virtual_instrument,
        inputs=[
            text2sound_sample_index_slider,
            text2sound_instrument_name_textbox,
            text2sound_state,
            virtual_instruments_state
        ],
        outputs=[
            virtual_instruments_state,
            text2sound_instrument_name_textbox
        ]
    )

    # Bind the event for updating the displayed sample when the sample index slider changes.
    text2sound_sample_index_slider.change(
        show_random_sample,
        inputs=[text2sound_sample_index_slider, text2sound_state],
        outputs=[
            text2sound_latent_representation_image,
            text2sound_quantized_latent_representation_image,
            text2sound_sampled_spectrogram_image,
            text2sound_sampled_phase_image,
            text2sound_sampled_audio
        ]
    )
