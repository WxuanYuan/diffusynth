import gradio as gr

readme_content = """## Assisting Musicians with Generation of Musical Notes using a Text-Guided Diffusion Model


### Training Data:
The neural network is trained on the filtered NSynth dataset [3], which comes with the following labels:

Instrument Families: bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, synth lead, vocal.
Instrument Sources: acoustic, electronic, synthetic.
Note Qualities: bright, dark, distortion, fast decay, long release, multiphonic, nonlinear env, percussive, reverb, tempo-synced.

YOU ARE NOT LIMITED TO THE ABOVE TERMS; THE MODEL CAN UNDERSTAND A WIDE RANGE OF VOCABULARY AND ACCEPTS NATURAL LANGUAGE INPUT!

### Usage Hints:

1. **Unique Sounds**: Start generating your unique sound in Text2Sound!

2. **Sample Indexing**: Drag the "Sample index slider" to view other samples within the generated batch.

3. **Editing Sounds**: Generated audio can be downloaded and re-uploaded for further editing in the Sound2Sound/Inpaint sections. YOU CAN ALSO UPLOAD OR RECORD AUDIO FROM OTHER SOURCES.

4. **Arrangement** Once you have achieved a satisfactory timbre in the Text2Sound, Sound2Sound, or Inpaint module, you can name and save it in the bottom-right corner. Then, you can upload your MIDI file in the Arrangement module, assign the saved timbre to each track, and start playing!

References:

[1] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10684-10695).

[2] AUTOMATIC1111. (2022). Stable Diffusion Web UI [Computer software]. Retrieved from https://github.com/AUTOMATIC1111/stable-diffusion-webui

[3] Engel, J., Resnick, C., Roberts, A., Dieleman, S., Eck, D., Simonyan, K., & Norouzi, M. (2017). Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders.

[4] Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598.
"""

def get_readme_module():

    with gr.Tab("README"):
        # gr.Markdown("Use interpolation to generate a gradient sound sequence.")
        with gr.Column(scale=3):
            readme_textbox = gr.Textbox(label="readme", lines=40, value=readme_content, interactive=False)