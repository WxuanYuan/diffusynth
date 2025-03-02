import torch
import numpy as np
from tqdm import tqdm
from model.DiffSynthSampler import DiffSynthSampler
import soundfile as sf
from tqdm import tqdm
from model.VQGAN import get_VQGAN
from model.diffusion import get_diffusion_model
from transformers import AutoTokenizer, ClapModel
from model.diffusion_components import linear_beta_schedule
from model.timbre_encoder_pretrain import get_timbre_encoder
from model.multimodal_model import get_multi_modal_model



import gradio as gr
from webUI.natural_language_guided_4.gradio_webUI import GradioWebUI
from webUI.natural_language_guided_4.load_presets import load_presets
from webUI.natural_language_guided_4.text2sound import get_text2sound_module
from webUI.natural_language_guided_4.sound2sound_with_text import get_sound2sound_with_text_module
from webUI.natural_language_guided_4.inpaint_with_text import get_inpaint_with_text_module
from webUI.natural_language_guided_4.note2music import get_arrangement_module
from webUI.natural_language_guided_4.README import get_readme_module



device = "cuda" if torch.cuda.is_available() else "cpu"
use_pretrained_CLAP = False

# load VQ-GAN
VAE_model_name = "24_1_2024-52_4x_L_D"
modelConfig = {"in_channels": 3, "hidden_channels": [80, 160], "embedding_dim": 4, "out_channels": 3, "block_depth": 2,
               "attn_pos":  [80, 160], "attn_with_skip": True,
            "num_embeddings": 8192, "commitment_cost": 0.25, "decay": 0.99,
            "norm_type": "groupnorm", "act_type": "swish", "num_groups": 16}
VAE = get_VQGAN(modelConfig, load_pretrain=True, model_name=VAE_model_name, device=device)

# load U-Net
UNet_model_name = "history/28_1_2024_CLAP_STFT_180000" if use_pretrained_CLAP else "history/28_1_2024_TE_STFT_300000"
unetConfig = {"in_dim": 4, "down_dims": [96, 96, 192, 384], "up_dims": [384, 384, 192, 96], "attn_type": "linear_add", "condition_type": "natural_language_prompt", "label_emb_dim": 512}
uNet = get_diffusion_model(unetConfig, load_pretrain=True, model_name=UNet_model_name, device=device)

# load LM
CLAP_temp = ClapModel.from_pretrained("laion/clap-htsat-unfused")
CLAP_tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

timbre_encoder_name = "24_1_2024_STFT"
timbre_encoder_Config = {"input_dim": 512, "feature_dim": 512, "hidden_dim": 1024, "num_instrument_classes": 1006, "num_instrument_family_classes": 11, "num_velocity_classes": 128, "num_qualities": 10, "num_layers": 3}
timbre_encoder = get_timbre_encoder(timbre_encoder_Config, load_pretrain=True, model_name=timbre_encoder_name, device=device)

if use_pretrained_CLAP:
  text_encoder = CLAP_temp
else:
  multimodalmodel_name = "24_1_2024"
  multimodalmodel_config = {"text_feature_dim": 512, "spectrogram_feature_dim": 1024, "multi_modal_emb_dim": 512, "num_projection_layers": 2,
                "temperature": 1.0, "dropout": 0.1, "freeze_text_encoder": False, "freeze_spectrogram_encoder": False}
  mmm = get_multi_modal_model(timbre_encoder, CLAP_temp, multimodalmodel_config, load_pretrain=True, model_name=multimodalmodel_name, device=device)

  text_encoder = mmm.to("cpu")





gradioWebUI = GradioWebUI(device, VAE, uNet, text_encoder, CLAP_tokenizer, freq_resolution=512, time_resolution=256, channels=4, timesteps=1000, squared=False,
                          VAE_scale=4, flexible_duration=True, noise_strategy="repeat", GAN_generator=None)

virtual_instruments, midis = load_presets(gradioWebUI)



with gr.Blocks(theme=gr.themes.Soft(), mode="dark") as demo:
    gr.Markdown("Thank you for using DiffuSynth v0.2!")

    reconstruction_state = gr.State(value={})
    text2sound_state = gr.State(value={})
    sound2sound_state = gr.State(value={})
    inpaint_state = gr.State(value={})
    super_resolution_state = gr.State(value={})
    virtual_instruments_state = gr.State(value={"virtual_instruments": virtual_instruments})
    midi_files_state = gr.State(value={"midis": midis})

    get_text2sound_module(gradioWebUI, text2sound_state, virtual_instruments_state)
    get_sound2sound_with_text_module(gradioWebUI, sound2sound_state, virtual_instruments_state)
    get_inpaint_with_text_module(gradioWebUI, inpaint_state, virtual_instruments_state)
    # get_build_instrument_module(gradioWebUI, virtual_instruments_state)
    get_arrangement_module(gradioWebUI, virtual_instruments_state, midi_files_state)
    get_readme_module()
    # get_instruments_module(gradioWebUI, virtual_instruments_state)

demo.launch(debug=True, share=True)
# demo.launch(debug=True, share=False)





















