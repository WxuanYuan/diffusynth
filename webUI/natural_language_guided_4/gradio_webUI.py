import gradio as gr


class GradioWebUI():
    """
    A class to build a Gradio-based web UI for sound generation and manipulation.
    This class holds the models, configuration parameters, and state variables required
    for interactive operations.
    """

    def __init__(self, device, VAE, uNet, CLAP, CLAP_tokenizer,
                 freq_resolution=512, time_resolution=256, channels=4, timesteps=1000,
                 sample_rate=16000, squared=False, VAE_scale=4,
                 flexible_duration=False, noise_strategy="repeat",
                 GAN_generator=None):
        """
        Initialize the GradioWebUI object with the provided models and parameters.

        Args:
            device (str): The device to use (e.g., "cpu" or "cuda").
            VAE: The VAE model object containing _encoder, _vq_vae, and _decoder.
            uNet: The uNet model for sound generation or processing.
            CLAP: The CLAP model for extracting text or sound features.
            CLAP_tokenizer: The tokenizer to use with the CLAP model.
            freq_resolution (int, optional): Frequency resolution. Default is 512.
            time_resolution (int, optional): Time resolution. Default is 256.
            channels (int, optional): Number of audio channels. Default is 4.
            timesteps (int, optional): Number of sampling steps. Default is 1000.
            sample_rate (int, optional): Audio sample rate. Default is 16000.
            squared (bool, optional): Whether to use squared mode. Default is False.
            VAE_scale (int, optional): Scale factor for the VAE model. Default is 4.
            flexible_duration (bool, optional): Allow flexible sound duration. Default is False.
            noise_strategy (str, optional): Strategy for noise generation. Default is "repeat".
            GAN_generator: Optional GAN generator. Default is None.
        """

        self.device = device
        self.VAE_encoder, self.VAE_quantizer, self.VAE_decoder = VAE._encoder, VAE._vq_vae, VAE._decoder
        self.uNet = uNet
        self.CLAP, self.CLAP_tokenizer = CLAP, CLAP_tokenizer
        self.freq_resolution, self.time_resolution = freq_resolution, time_resolution
        self.channels = channels
        self.GAN_generator = GAN_generator

        self.timesteps = timesteps
        self.sample_rate = sample_rate
        self.squared = squared
        self.VAE_scale = VAE_scale
        self.flexible_duration = flexible_duration
        self.noise_strategy = noise_strategy

        self.text2sound_state = gr.State(value={})
        self.interpolation_state = gr.State(value={})
        self.sound2sound_state = gr.State(value={})
        self.inpaint_state = gr.State(value={})

    def get_sample_steps_slider(self):
        default_steps = 10 if (self.device == "cpu") else 20
        return gr.Slider(minimum=10, maximum=100, value=default_steps, step=1,
                         label="Sample steps",
                         info="Sampling steps. The more sampling steps, the better the "
                              "theoretical result, but the time it consumes.")

    def get_sampler_radio(self):
        # return gr.Radio(choices=["ddpm", "ddim", "dpmsolver++", "dpmsolver"], value="ddim", label="Sampler")
        return gr.Radio(choices=["ddpm", "ddim"], value="ddim", label="Sampler")

    def get_batchsize_slider(self, cpu_batchsize=1):
        return gr.Slider(minimum=1., maximum=16, value=cpu_batchsize if (self.device == "cpu") else 8, step=1, label="Batchsize")

    def get_time_resolution_slider(self):
        return gr.Slider(minimum=16., maximum=int(1024/self.VAE_scale), value=int(256/self.VAE_scale), step=1, label="Time resolution", interactive=True)

    def get_duration_slider(self):
        if self.flexible_duration:
            return gr.Slider(minimum=0.25, maximum=8., value=3., step=0.01, label="duration in sec")
        else:
            return gr.Slider(minimum=1., maximum=8., value=3., step=1., label="duration in sec")

    def get_guidance_scale_slider(self):
        return gr.Slider(minimum=0., maximum=20., value=6., step=1.,
                         label="Guidance scale",
                         info="The larger this value, the more the generated sound is "
                              "influenced by the condition. Setting it to 0 is equivalent to "
                              "the negative case.")

    def get_noising_strength_slider(self, default_noising_strength=0.7):
        return gr.Slider(minimum=0.0, maximum=1.00, value=default_noising_strength, step=0.01,
                         label="noising strength",
                         info="The smaller this value, the more the generated sound is "
                              "closed to the origin.")

    def get_seed_textbox(self):
        return gr.Textbox(label="Seed", lines=1, placeholder="seed", value=0)
