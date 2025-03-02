import torch
import gradio as gr
import mido
from io import BytesIO
# import pyrubberband as pyrb

from webUI.natural_language_guided_4.track_maker import DiffSynth, Track


def get_arrangement_module(gradioWebUI, virtual_instruments_state, midi_files_state):
    """
    Build the arrangement module which allows users to generate a full track based on
    selected MIDI files and virtual instruments. This module handles MIDI file reading,
    instrument selection for each track, and track generation using DiffSynth.

    Args:
        gradioWebUI: The main Gradio UI object containing model configurations.
        virtual_instruments_state: The Gradio state containing virtual instrument information.
        midi_files_state: The Gradio state containing loaded MIDI files.

    Returns:
        None. (The function builds UI components and binds events.)
    """
    # Load configurations from gradioWebUI
    uNet = gradioWebUI.uNet
    freq_resolution, time_resolution = gradioWebUI.freq_resolution, gradioWebUI.time_resolution
    VAE_scale = gradioWebUI.VAE_scale
    height, width, channels = int(freq_resolution / VAE_scale), int(time_resolution / VAE_scale), gradioWebUI.channels

    timesteps = gradioWebUI.timesteps
    VAE_quantizer = gradioWebUI.VAE_quantizer
    VAE_decoder = gradioWebUI.VAE_decoder
    CLAP = gradioWebUI.CLAP
    CLAP_tokenizer = gradioWebUI.CLAP_tokenizer
    device = gradioWebUI.device
    squared = gradioWebUI.squared
    sample_rate = gradioWebUI.sample_rate
    noise_strategy = gradioWebUI.noise_strategy

    def read_midi(midi, midi_dict):
        """
        Read an uploaded MIDI file and update the MIDI state.

        Args:
            midi: The uploaded MIDI file in binary format.
            midi_dict: The current dictionary of MIDI files.

        Returns:
            A dictionary with:
                - A Textbox component displaying MIDI info.
                - An updated current MIDI state.
                - The updated midi_files_state dictionary.
        """
        # Read the MIDI file from binary data using BytesIO
        mid = mido.MidiFile(file=BytesIO(midi))
        # Create Track objects for each track in the MIDI file
        tracks = [Track(t, mid.ticks_per_beat) for t in mid.tracks]

        # Build a string with information about the loaded MIDI
        midi_info_text = f"Uploaded midi:"
        for i, track in enumerate(tracks):
            midi_info_text += f"\n{len(track.events)} events loaded from Track {i}."

        # Update the MIDI state dictionary with the uploaded MIDI file
        midis = midi_dict["midis"]
        midis["uploaded_midi"] = mid
        midi_dict["midis"] = midis

        # Return a dictionary updating the MIDI info textbox, current MIDI state, and MIDI state dictionary
        return {
            midi_info_textbox: gr.Textbox(label="Midi info", lines=10, placeholder=midi_info_text),
            current_midi_state: "uploaded_midi",
            midi_files_state: midi_dict
        }

    def make_track(inpaint_steps, current_midi_name, midi_dict, max_notes, noising_strength, attack, before_release,
                   current_instruments, virtual_instruments_dict):
        """
        Generate a full track based on the selected MIDI file and instrument configurations.

        Args:
            inpaint_steps (float): The number of inpainting steps (sampling steps).
            current_midi_name (str): The name of the currently selected MIDI file.
            midi_dict (dict): The dictionary containing loaded MIDI files.
            max_notes (float): The maximum number of synthesized notes per track.
            noising_strength (float): The noising strength parameter.
            attack (float): Attack duration parameter.
            before_release (float): Time before release parameter.
            current_instruments (list): A list of instrument names selected for each track.
            virtual_instruments_dict (dict): Dictionary of available virtual instruments.

        Returns:
            dict: A dictionary with the generated track audio.
        """
        # Warn if noising strength is less than 1 (may affect quality)
        if noising_strength < 1:
            print(f"Warning: making track with noising_strength = {noising_strength} < 1")
        virtual_instruments = virtual_instruments_dict["virtual_instruments"]
        sample_steps = int(inpaint_steps)

        print(f"current_instruments: {current_instruments}")
        # Use the provided instrument names for each track
        instrument_names = current_instruments
        instruments_configs = {}

        # For each selected instrument, prepare its configuration for track generation
        for virtual_instrument_name in instrument_names:
            virtual_instrument = virtual_instruments[virtual_instrument_name]

            # Convert the instrument's latent representation to a tensor and move to device
            latent_representation = torch.tensor(virtual_instrument["latent_representation"], dtype=torch.float32).to(
                device)
            sampler = virtual_instrument["sampler"]

            batchsize = 1

            # Repeat the latent representation to match the batch size
            latent_representation = latent_representation.repeat(batchsize, 1, 1, 1)

            instruments_configs[virtual_instrument_name] = {
                'sample_steps': sample_steps,
                'sampler': sampler,
                'noising_strength': noising_strength,
                'latent_representation': latent_representation,
                'attack': attack,
                'before_release': before_release
            }

        # Create a DiffSynth object to generate music
        diffSynth = DiffSynth(instruments_configs, uNet, VAE_quantizer, VAE_decoder, CLAP, CLAP_tokenizer, device)

        # Retrieve the selected MIDI file from the state dictionary
        midis = midi_dict["midis"]
        mid = midis[current_midi_name]
        # Generate the full audio track using the selected instruments and MIDI file
        full_audio = diffSynth.get_music(mid, instrument_names, max_notes=max_notes)

        # Return a dictionary updating the track audio component with the generated sound
        return {track_audio: (sample_rate, full_audio)}

    # Build the UI layout inside a Gradio Tab labeled "Arrangement"
    with gr.Tab("Arrangement"):
        # Set default instrument and current MIDI state
        default_instrument = "preset_string"
        current_instruments_state = gr.State(value=[default_instrument for _ in range(100)])
        current_midi_state = gr.State(value="Ode_to_Joy_Easy_variation")

        gr.Markdown("Make music with generated sounds!")
        with gr.Row(variant="panel"):
            with gr.Column(scale=3):
                # Use a render decorator to create a dynamic dropdown for preset MIDI files
                @gr.render(inputs=midi_files_state)
                def check_midis(midi_dict):
                    midis = midi_dict["midis"]
                    midi_names = list(midis.keys())

                    instrument_dropdown = gr.Dropdown(
                        midi_names, label="Select from preset midi files", value="Ode_to_Joy_Easy_variation"
                    )

                    def select_midi(midi_name):
                        # For the selected MIDI, create Track objects and prepare MIDI info text
                        mid = midis[midi_name]
                        tracks = [Track(t, mid.ticks_per_beat) for t in mid.tracks]
                        midi_info_text = f"Name: {midi_name}"
                        for i, track in enumerate(tracks):
                            midi_info_text += f"\n{len(track.events)} events loaded from Track {i}."
                        # Return updates for the MIDI info textbox and current MIDI state
                        return {
                            midi_info_textbox: gr.Textbox(label="Midi info", lines=10, placeholder=midi_info_text),
                            current_midi_state: midi_name
                        }

                    instrument_dropdown.select(select_midi, inputs=instrument_dropdown,
                                               outputs=[midi_info_textbox, current_midi_state])

                # File component for uploading a MIDI file
                midi_file = gr.File(label="Upload a midi file", type="binary", scale=1)
                midi_info_textbox = gr.Textbox(
                    label="Midi info", lines=10,
                    placeholder="Please select/upload a midi on the left.",
                    scale=3,
                    visible=False
                )

            with gr.Column(scale=3):
                # Render dynamic instrument selection for each track based on the current MIDI file
                @gr.render(inputs=[current_midi_state, midi_files_state, virtual_instruments_state])
                def render_select_instruments(current_midi_name, midi_dict, virtual_instruments_dict):
                    virtual_instruments = virtual_instruments_dict["virtual_instruments"]
                    instrument_names = list(virtual_instruments.keys())

                    midis = midi_dict["midis"]
                    mid = midis[current_midi_name]
                    tracks = [Track(t, mid.ticks_per_beat) for t in mid.tracks]

                    dropdowns = []
                    # Create a dropdown for each track in the MIDI file
                    for i, track in enumerate(tracks):
                        dropdowns.append(gr.Dropdown(
                            instrument_names,
                            value=default_instrument,
                            label=f"Track {i}:  {len(track.events)} notes",
                            info="Select an instrument to play this track!"
                        ))

                    # When instruments are selected, update the current instruments state
                    def select_instruments(*instruments):
                        return instruments

                    for d in dropdowns:
                        d.select(select_instruments, inputs=dropdowns, outputs=current_instruments_state)

            with gr.Column(scale=3):
                # Slider for maximum number of synthesized notes per track
                max_notes_slider = gr.Slider(
                    minimum=10.0, maximum=999.0, value=100.0, step=1.0,
                    label="Maximum number of synthesized notes in each track",
                    info="Lower this value to prevent Gradio timeouts"
                )
                # Button to trigger track generation
                make_track_button = gr.Button(variant="primary", value="Make track", scale=1)
                # Audio component to play the generated track
                track_audio = gr.Audio(type="numpy", label="Play music", interactive=False)

        # Additional configuration panels (hidden by default) for advanced settings
        with gr.Row(variant="panel", visible=False):
            with gr.Tab("Origin sound"):
                inpaint_steps_slider = gr.Slider(
                    minimum=5.0, maximum=999.0, value=20.0, step=1.0,
                    label="inpaint_steps"
                )
                noising_strength_slider = gradioWebUI.get_noising_strength_slider(default_noising_strength=1.)
                end_noise_level_ratio_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.01,
                    label="end_noise_level_ratio"
                )
                attack_slider = gr.Slider(
                    minimum=0.0, maximum=1.5, value=0.5, step=0.01,
                    label="attack in sec"
                )
                before_release_slider = gr.Slider(
                    minimum=0.0, maximum=1.5, value=0.5, step=0.01,
                    label="before_release in sec"
                )
                release_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.3, step=0.01,
                    label="release in sec"
                )
                mask_flexivity_slider = gr.Slider(
                    minimum=0.01, maximum=1.0, value=1.0, step=0.01,
                    label="mask_flexivity"
                )
            with gr.Tab("Length adjustment config"):
                use_dynamic_mask_checkbox = gr.Checkbox(label="Use dynamic mask", value=True)
                test_duration_envelope_button = gr.Button(variant="primary", value="Apply envelope", scale=1)
                test_duration_stretch_button = gr.Button(variant="primary", value="Apply stretch", scale=1)
                test_duration_inpaint_button = gr.Button(variant="primary", value="Inpaint different duration", scale=1)
                duration_slider = gradioWebUI.get_duration_slider()
            with gr.Tab("Pitch shift config"):
                pitch_shift_radio = gr.Radio(
                    choices=["librosa", "torchaudio", "rubberband"],
                    value="librosa"
                )

        with gr.Row(variant="panel", visible=False):
            with gr.Column(scale=2):
                with gr.Row(variant="panel"):
                    source_sound_spectrogram_image = gr.Image(
                        label="New sound spectrogram", type="numpy",
                        height=600, scale=1
                    )
                    source_sound_phase_image = gr.Image(
                        label="New sound phase", type="numpy",
                        height=600, scale=1
                    )

        # Bind the event for the "Make track" button click to generate the track
        make_track_button.click(
            make_track,
            inputs=[
                inpaint_steps_slider, current_midi_state, midi_files_state,
                max_notes_slider, noising_strength_slider,
                attack_slider, before_release_slider,
                current_instruments_state, virtual_instruments_state
            ],
            outputs=[track_audio]
        )

        # Bind the event for when a MIDI file is uploaded, to process and update the MIDI state
        midi_file.change(
            read_midi,
            inputs=[midi_file, midi_files_state],
            outputs=[midi_info_textbox, current_midi_state, midi_files_state]
        )

    # End of the arrangement module.
