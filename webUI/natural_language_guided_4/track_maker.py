import librosa
import numpy as np
import torch

from model.DiffSynthSampler import DiffSynthSampler
from webUI.natural_language_guided_4.utils import encodeBatch2GradioOutput_STFT
import mido
import torchaudio.transforms as transforms
from tqdm import tqdm


def pitch_shift_librosa(waveform, sample_rate, total_steps, step_size=4, n_fft=4096, hop_length=None):
    """
    Shift the pitch of an audio waveform using librosa in incremental steps.

    Args:
        waveform (numpy.ndarray or torch.Tensor): The input waveform.
        sample_rate (int): The sample rate of the audio.
        total_steps (int): The total number of semitones to shift.
        step_size (int): The number of semitones to shift per iteration (default: 4).
        n_fft (int): FFT window size (default: 4096).
        hop_length (int, optional): Hop length for STFT; if None, defaults to n_fft/4.

    Returns:
        numpy.ndarray: The pitch-shifted waveform.
    """
    # Ensure the waveform is a numpy array
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    # Set default hop_length if not provided
    if hop_length is None:
        hop_length = n_fft // 4

    # Perform pitch shifting in increments
    current_waveform = waveform
    num_steps = int(np.ceil(total_steps / step_size))

    for i in range(num_steps):
        # Ensure that the final shift does not exceed the total_steps
        step = min(step_size, total_steps - i * step_size)
        current_waveform = librosa.effects.pitch_shift(
            current_waveform, sr=sample_rate, n_steps=step,
            n_fft=n_fft, hop_length=hop_length
        )

    return current_waveform


class NoteEvent:
    """
    A class representing a MIDI note event.
    """

    def __init__(self, note, velocity, start_time, duration):
        self.note = note  # MIDI note number
        self.velocity = velocity  # Velocity of the note
        self.start_time = start_time  # Start time in ticks
        self.duration = duration  # Duration in ticks

    def __str__(self):
        return f"Note {self.note}, velocity {self.velocity}, start_time {self.start_time}, duration {self.duration}"


class Track:
    """
    Represents a MIDI track and provides functionality to parse note and tempo events
    and synthesize the track into an audio waveform.
    """

    def __init__(self, track, ticks_per_beat, max_notes=100):
        # Parse tempo events and note events from the track
        self.tempo_events = self._parse_tempo_events(track)
        self.events = self._parse_note_events(track)
        self.ticks_per_beat = ticks_per_beat
        self.max_notes = int(max_notes)

    def _parse_tempo_events(self, track):
        """
        Parse tempo events from the MIDI track.

        Returns:
            list: A list of (time, tempo) tuples.
        """
        tempo_events = []
        current_tempo = 500000  # Default tempo: 500,000 microseconds per beat (120 BPM)
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_events.append((msg.time, msg.tempo))
            elif not msg.is_meta:
                # For non-meta messages, assume the current tempo
                tempo_events.append((msg.time, current_tempo))
        return tempo_events

    def _parse_note_events(self, track):
        """
        Parse note events from the MIDI track.

        Returns:
            list: A list of NoteEvent objects.
        """
        events = []
        start_time = 0
        for msg in track:
            if not msg.is_meta:
                start_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_on_time = start_time
                elif msg.type == 'note_on' and msg.velocity == 0:
                    duration = start_time - note_on_time
                    events.append(NoteEvent(msg.note, msg.velocity, note_on_time, duration))
        return events

    def synthesize_track(self, diffSynthSampler, sample_rate=16000):
        """
        Synthesize audio for this track using a provided DiffSynthSampler.

        Args:
            diffSynthSampler: A function that generates an audio sample for a note.
            sample_rate (int): The target sample rate.

        Returns:
            numpy.ndarray: The synthesized audio waveform for the track.
        """
        # Calculate total duration of the track in seconds and initialize the audio array
        track_audio = np.zeros(int(self._get_total_time() * sample_rate), dtype=np.float32)
        current_tempo = 500000  # Start with default tempo
        duration_note_mapping = {}

        # Process each note event (up to max_notes)
        for event in tqdm(self.events[:self.max_notes]):
            current_tempo = self._get_tempo_at(event.start_time)
            seconds_per_tick = mido.tick2second(1, self.ticks_per_beat, current_tempo)
            start_time_sec = event.start_time * seconds_per_tick
            # Ensure a minimum note duration of 0.75 seconds
            duration_sec = max(event.duration * seconds_per_tick, 0.75)
            start_sample = int(start_time_sec * sample_rate)
            # Generate the note sample if not already generated for this duration
            if str(duration_sec) not in duration_note_mapping:
                note_sample = diffSynthSampler(event.velocity, duration_sec)
                # Normalize the note sample
                duration_note_mapping[str(duration_sec)] = note_sample / np.max(np.abs(note_sample))
            # Apply pitch shifting based on the note (difference from a reference note, e.g., 52)
            note_audio = pitch_shift_librosa(duration_note_mapping[str(duration_sec)], sample_rate, event.note - 52)
            end_sample = start_sample + len(note_audio)
            # Mix the note audio into the track audio
            track_audio[start_sample:end_sample] += note_audio

        return track_audio

    def _get_tempo_at(self, time_tick):
        """
        Determine the tempo at a given tick in the track.

        Args:
            time_tick (int): The tick at which to determine the tempo.

        Returns:
            int: The tempo (in microseconds per beat) at the specified tick.
        """
        current_tempo = 500000  # Default tempo
        elapsed_ticks = 0

        for tempo_change in self.tempo_events:
            if elapsed_ticks + tempo_change[0] > time_tick:
                return current_tempo
            elapsed_ticks += tempo_change[0]
            current_tempo = tempo_change[1]

        return current_tempo

    def _get_total_time(self):
        """
        Compute the total time of the track in seconds.

        Returns:
            float: Total duration of the track (plus an extra 10 seconds for safety).
        """
        total_time = 0
        current_tempo = 500000  # Default tempo

        for event in self.events:
            current_tempo = self._get_tempo_at(event.start_time)
            seconds_per_tick = mido.tick2second(1, self.ticks_per_beat, current_tempo)
            total_time += event.duration * seconds_per_tick

        return total_time + 10


class DiffSynth:
    """
    DiffSynth synthesizes audio using a diffusion-based inpainting model.

    It uses instrument configurations to guide the inpainting process, producing new
    audio samples that can be combined to generate full music tracks.
    """

    def __init__(self, instruments_configs, noise_prediction_model, VAE_quantizer, VAE_decoder, text_encoder,
                 CLAP_tokenizer, device,
                 model_sample_rate=16000, timesteps=1000, channels=4, freq_resolution=512, time_resolution=256,
                 VAE_scale=4, squared=False):

        self.noise_prediction_model = noise_prediction_model
        self.VAE_quantizer = VAE_quantizer
        self.VAE_decoder = VAE_decoder
        self.device = device
        self.model_sample_rate = model_sample_rate
        self.timesteps = timesteps
        self.channels = channels
        self.freq_resolution = freq_resolution
        self.time_resolution = time_resolution
        self.height = int(freq_resolution / VAE_scale)
        self.VAE_scale = VAE_scale
        self.squared = squared
        self.text_encoder = text_encoder
        self.CLAP_tokenizer = CLAP_tokenizer

        # instruments_configs is a dictionary mapping instrument names to their configurations.
        self.instruments_configs = instruments_configs
        self.diffSynthSamplers = {}
        self._update_instruments()

    def _update_instruments(self):
        """
        For each instrument configuration, create a DiffSynthSampler wrapper function.
        """

        def diffSynthSamplerWrapper(instruments_config):
            def diffSynthSampler(velocity, duration_sec, sample_rate=16000):
                # Obtain a default condition from the text encoder using an empty prompt.
                condition = self.text_encoder.get_text_features(
                    **self.CLAP_tokenizer([""], padding=True, return_tensors="pt")
                ).to(self.device)
                sample_steps = instruments_config['sample_steps']
                sampler = instruments_config['sampler']
                noising_strength = instruments_config['noising_strength']
                latent_representation = instruments_config['latent_representation']
                attack = instruments_config['attack']
                before_release = instruments_config['before_release']

                # Ensure the sample rate matches the model sample rate.
                assert sample_rate == self.model_sample_rate, "sample_rate != model_sample_rate"

                # Calculate the width of the latent space representation based on duration.
                width = int(self.time_resolution * ((duration_sec + 1) / 4) / self.VAE_scale)

                # Initialize a DiffSynthSampler with fixed height and channels.
                mySampler = DiffSynthSampler(self.timesteps, height=128, channels=4, noise_strategy="repeat", mute=True)
                mySampler.respace(list(np.linspace(0, self.timesteps - 1, sample_steps, dtype=np.int32)))

                # Create a latent mask to "freeze" certain parts of the latent representation.
                latent_mask = torch.zeros((1, 1, self.height, width), dtype=torch.float32).to(self.device)
                latent_mask[:, :, :, :int(self.time_resolution * (attack / 4) / self.VAE_scale)] = 1.0
                latent_mask[:, :, :, -int(self.time_resolution * ((before_release + 1) / 4) / self.VAE_scale):] = 1.0

                # Perform inpainting sampling using the noise prediction model.
                latent_representations, _ = mySampler.inpaint_sample(
                    model=self.noise_prediction_model,
                    shape=(1, self.channels, self.height, width),
                    noising_strength=noising_strength,
                    condition=condition,
                    guide_img=latent_representation,
                    mask=latent_mask,
                    return_tensor=True,
                    sampler=sampler,
                    use_dynamic_mask=True,
                    end_noise_level_ratio=0.0,
                    mask_flexivity=1.0
                )

                # Select the final latent representation.
                latent_representations = latent_representations[-1]

                # Quantize the latent representation.
                quantized_latent_representations, _, (_, _, _) = self.VAE_quantizer(latent_representations)
                # Decode the quantized representation to obtain spectrograms, phase images, and audio.
                flipped_log_spectrums, flipped_phases, rec_signals, _, _, _ = encodeBatch2GradioOutput_STFT(
                    self.VAE_decoder,
                    quantized_latent_representations,
                    resolution=(512, width * self.VAE_scale),
                    original_STFT_batch=None,
                )
                return rec_signals[0]

            return diffSynthSampler

        # Create a sampler for each instrument configuration and store it.
        for key in self.instruments_configs.keys():
            self.diffSynthSamplers[key] = diffSynthSamplerWrapper(self.instruments_configs[key])

    def get_music(self, mid, instrument_names, sample_rate=16000, max_notes=100):
        """
        Generate full audio by synthesizing each MIDI track using the corresponding instrument.

        Args:
            mid: A mido.MidiFile object representing the MIDI file.
            instrument_names (list): List of instrument names to use for each track.
            sample_rate (int): The target sample rate.
            max_notes (int): Maximum number of notes to synthesize per track.

        Returns:
            numpy.ndarray: The synthesized full audio by summing all tracks.
        """
        # Create Track objects for each track in the MIDI file.
        tracks = [Track(t, mid.ticks_per_beat, max_notes) for t in mid.tracks]
        # Ensure there are not more tracks than instrument names.
        assert len(tracks) <= len(
            instrument_names), f"len(tracks) = {len(tracks)} > {len(instrument_names)} = len(instrument_names)"

        # Synthesize audio for each track using the corresponding instrument's sampler.
        track_audios = [
            track.synthesize_track(self.diffSynthSamplers[instrument_names[i]], sample_rate=sample_rate)
            for i, track in enumerate(tracks)
        ]

        # Pad all track audios to the length of the longest track.
        max_length = max(len(audio) for audio in track_audios)
        full_audio = np.zeros(max_length, dtype=np.float32)  # Initialize full audio array with zeros.
        for audio in track_audios:
            padded_audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
            full_audio += padded_audio  # Sum the padded audios.
        return full_audio
