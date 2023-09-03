import gradio as gr
from gtts import gTTS
import subprocess
from datetime import datetime
import torchaudio
import torch
import pysrt
import shutil
import os
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voices

def rest_subs(end_sub, start_sub):
    end_sub = datetime.combine(datetime.today(), end_sub.to_time())
    start_sub = datetime.combine(datetime.today(), start_sub.to_time())
    duration = (end_sub - start_sub).total_seconds()
    return duration

# gtts model
# def t2v(text, name):
#     tts = gTTS(text, lang='en')
#     tts.save(f'results/tmp/{name}.2.wav')

# tortoise model
def t2v(tts, emotion, text, voice_samples, custom_preset, name):
    gen = tts.tts_custom(emotion+text, voice_samples=voice_samples, custom_preset=custom_preset)
    torchaudio.save(f'results/tmp/{name}.2.wav', gen.squeeze(0).cpu(), 24000)

def save_silent(duration, name):
    # Compatible with gtts (replace all .wav to .wav)
    # command = ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(duration), "-acodec", "libmp3lame", f"results/tmp/{name}.wav"]
    # compatible with tortoise (replace all .wav to .wav)
    command = ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=24000", "-t", str(duration), "-c:a", "pcm_f32le", f"results/tmp/{name}.wav"]
    subprocess.run(command, capture_output=True)
    print(f"Silence: {duration}s")

def shrink_audio(duration, name):
    command = f"ffprobe -i results/tmp/{name}.2.wav -show_entries format=duration -v quiet -of csv=\"p=0\" -loglevel quiet"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_p = process.communicate()[0].decode("utf-8")
    duration1 = float(output_p.strip())

    if duration1 > duration:
        atempo = duration1 / duration
        print(f"Accelerate by {round(atempo, 2)}x")
        command = f"ffmpeg -i results/tmp/{name}.2.wav -filter:a \"atempo={atempo}\" -vn -loglevel quiet results/tmp/{name}.2_.wav"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output_p = process.communicate()[0].decode("utf-8")
        os.remove(f'results/tmp/{name}.2.wav')
        os.rename(f'results/tmp/{name}.2_.wav', f'results/tmp/{name}.2.wav')
    else:
        atempo = duration1 / duration
        save_silent(duration - duration1, f'{name}.3')

def file_key(file):
    parts = file.split('.')
    return int(parts[0])

##########################################################################

def refresh_btn_click():
    global voices_list
    voices_list = os.listdir('tortoise/voices')
    return gr.Dropdown.update(choices=voices_list, value=voices_list[0])

def generate_btn_click(subtitle, voice, emotion, num_autoregressive_samples, diffusion_iterations, temperature, length_penalty, repetition_penalty, top_p, cond_free, cond_free_k, diffusion_temperature, half):
    custom_preset = {
        'temperature': float(temperature),
        'length_penalty': float(length_penalty),
        'repetition_penalty': float(repetition_penalty),
        'top_p': float(top_p),
        'cond_free_k': float(cond_free_k),
        'diffusion_temperature': float(diffusion_temperature),
        'num_autoregressive_samples': num_autoregressive_samples,
        'diffusion_iterations': diffusion_iterations,
        'cond_free': cond_free
        }

    voice_samples, conditioning_latents = load_voices([voice])
    tts = TextToSpeech(half=half)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(sub_path, "wb") as f:
        f.write(subtitle)
    subs = pysrt.open(sub_path)

    start = pysrt.SubRipTime.from_ordinal(0)
    for number, sub in enumerate(subs):
        print(f" --> Processing subtitle {number}")
        duration = rest_subs(sub.start, start) # a duration
        save_silent(duration, f'{number}.1')
        t2v(tts, emotion, sub.text, voice_samples, custom_preset, number)
        #t2v(sub.text, number)
        duration = rest_subs(sub.end, sub.start) # audio duration
        start = sub.end
        shrink_audio(duration, number)
    torch.cuda.empty_cache()

    # Get the list of audio files in the directory and sort them
    files = [f for f in os.listdir(directory) if f.endswith(".wav")]
    files = sorted(files, key=file_key)

    # Create a command for ffmpeg to join the files
    command = ["ffmpeg", "-loglevel", "quiet", "-y"]
    for f in files:
        command.extend(["-i", os.path.join(directory, f)])
    command.extend(["-filter_complex", "concat=n={}:v=0:a=1".format(len(files)), output_path])
    subprocess.run(command)

    shutil.rmtree(directory)
    return gr.Audio.update(output_path)

##########################################################################

directory = "results/tmp/"
output_path = "results/output.wav"
sub_path = "results/tmp/sub.srt"
# list all the directories in 'tortoise/voices'
voices_list = os.listdir('tortoise/voices')
emotion_list = [None, '[I am really happy,] ', '[I am really sad,] ', '[I am really angry,] ', '[I am really disgusted,] ', '[Arrogant tone,] ', '[I am really surprised!,] ']

with gr.Blocks(title='ibarcena.net') as app:
    html = '''
        <a href='https://ibarcena.net/me'>
            <img src='https://ibarcena.net/content/images/2023/08/io2b-1.png alt='ibarcena.net/me'>
        </a>
    '''
    gr.HTML(html)

    with gr.Row():
        with gr.Column():
            subtitle = gr.File(type='binary', label='Subtitles file (.srt)')
            voice = gr.Dropdown(voices_list, value=voices_list[0], label='Voice')
            emotion = gr.Dropdown(emotion_list, value=emotion_list[0], label='Emotion')
            refresh_btn = gr.Button('Refresh')
        with gr.Column():
            num_autoregressive_samples = gr.Slider(minimum=4, maximum=512, value=4, label='Autoregressive samples', step=1)
            diffusion_iterations = gr.Slider(minimum=4, maximum=512, value=80, label='Diffusion iterations', step=1)
            temperature = gr.Slider(minimum=0.1, maximum=1, value=0.8, label='Temperature')
        with gr.Column():
            with gr.Row():
                half = gr.Checkbox(label='Half precision', value=True)
                cond_free = gr.Checkbox(label='Conditional free', value=True)
            cond_free_k = gr.Slider(minimum=0.1, maximum=4, value=2.0, label='Conditional free k')
            diffusion_temperature = gr.Slider(minimum=0.1, maximum=1, value=1.0, label='Diffusion temperature')
            length_penalty = gr.Slider(minimum=0.1, maximum=8, value=1.0, label='Length penalty')
            repetition_penalty = gr.Slider(minimum=0.1, maximum=8, value=2.0, label='Repetition penalty')
            top_p = gr.Slider(minimum=0.1, maximum=1, value=0.8, label='Top p')
        with gr.Column():
            generate_btn = gr.Button('Generate')
            output = gr.Audio(label='Output')

    refresh_btn.click(refresh_btn_click, outputs=[voice])
    generate_btn.click(generate_btn_click, [subtitle, voice, emotion, num_autoregressive_samples, diffusion_iterations, temperature, length_penalty, repetition_penalty, top_p, cond_free, cond_free_k, diffusion_temperature, half], outputs=[output])
    app.launch(share=False, debug=True)
