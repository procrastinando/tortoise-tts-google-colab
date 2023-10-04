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
from TTS.api import TTS
import random
import time

def rest_subs(end_sub, start_sub):
    end_sub = datetime.combine(datetime.today(), end_sub.to_time())
    start_sub = datetime.combine(datetime.today(), start_sub.to_time())
    duration = (end_sub - start_sub).total_seconds()
    return duration

# gtts model
def t2v_gtts(text, voice, name, extension):
    tts = gTTS(text, lang=voices_gtts[voice])
    tts.save(f'results/tmp/{name}.2.{extension}')

# coqui model
def t2v_coqui(tts, text, name, extension):
    tts.tts_to_file(text=text, file_path=f'results/tmp/{name}.2.{extension}')

# tortoise model
def t2v_tortoise(tts, emotion, text, voice_samples, custom_preset, name, extension):
    gen = tts.tts_custom(emotion+text, voice_samples=voice_samples, custom_preset=custom_preset)
    torchaudio.save(f'results/tmp/{name}.2.{extension}', gen.squeeze(0).cpu(), 24000)

def save_silent(duration, name, model, extension):
    if model == 'gtts':
        command = ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(duration), "-acodec", "libmp3lame", f"results/tmp/{name}.{extension}"]
    if model == 'coqui':
        command = ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono", "-t", str(duration), "-acodec", "libmp3lame", f"results/tmp/{name}.{extension}"]
    if model == 'tortoise':
        command = ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=24000", "-t", str(duration), "-c:a", "pcm_f32le", f"results/tmp/{name}.{extension}"]
    subprocess.run(command, capture_output=True)
    print(f"Silence: {duration}s")

def shrink_audio(duration, name, model, extension):
    command = f"ffprobe -i results/tmp/{name}.2.{extension} -show_entries format=duration -v quiet -of csv=\"p=0\" -loglevel quiet"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_p = process.communicate()[0].decode("utf-8")
    duration1 = float(output_p.strip())

    if duration1 > duration:
        atempo = duration1 / duration
        print(f"Accelerate by {round(atempo, 2)}x")
        command = f"ffmpeg -i results/tmp/{name}.2.{extension} -filter:a \"atempo={atempo}\" -vn -loglevel quiet results/tmp/{name}.2_.{extension}"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output_p = process.communicate()[0].decode("utf-8")
        os.remove(f'results/tmp/{name}.2.{extension}')
        os.rename(f'results/tmp/{name}.2_.{extension}', f'results/tmp/{name}.2.{extension}')
    else:
        save_silent(duration - duration1, f'{name}.3', model, extension)

def file_key(file):
    parts = file.split('.')
    return int(parts[0])

##########################################################################

def model_change(model):
    if model == 'gtts':
        voice = gr.Dropdown.update(choices=list(voices_gtts.keys()), value=list(voices_gtts.keys())[10])
        emotion = gr.Dropdown.update(visible=False)
        preview = gr.Audio.update(visible=False)
        refresh_btn = gr.Button.update(visible=False)
        params = gr.Column.update(visible=False)

    elif model == 'coqui':
        print(voices_coqui)
        voice = gr.Dropdown.update(choices=voices_coqui, value=voices_coqui[8])
        emotion = gr.Dropdown.update(visible=False)
        preview = gr.Audio.update(visible=False)
        refresh_btn = gr.Button.update(visible=False)
        params = gr.Column.update(visible=False)

    elif model == 'tortoise':
        voice = gr.Dropdown.update(choices=voices_tortoise, value=voices_tortoise[0])
        emotion = gr.Dropdown.update(visible=True)
        preview = gr.Audio.update(visible=True)
        refresh_btn = gr.Button.update(visible=True)
        params = gr.Column.update(visible=True)

    return voice, emotion, preview, refresh_btn, params

def voice_change(model, voice):
    if model == 'tortoise':
        return gr.Audio.update(f"tortoise/voices/{voice}/" + random.choice(os.listdir(f"tortoise/voices/{voice}/")))
    else:
        return gr.Audio.update()

def refresh_btn_click():
    global voices_tortoise
    voices_tortoise = os.listdir('tortoise/voices')
    return gr.Dropdown.update(choices=voices_tortoise, value=voices_tortoise[0])

def generate_btn_click(subtitle, voice, emotion, model, num_autoregressive_samples, diffusion_iterations, temperature, length_penalty, repetition_penalty, top_p, cond_free, cond_free_k, diffusion_temperature, half):
    start_time = time.time()
    # Clean temporary files
    if os.path.exists("results/"):
        shutil.rmtree("results/")
    os.makedirs('results/')
    os.makedirs('results/tmp/')

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

    with open("results/tmp/sub.srt", "wb") as f:
        f.write(subtitle)
    subs = pysrt.open("results/tmp/sub.srt")

    if model == 'gtts':
        extension = 'mp3'
    if model == 'coqui':
        extension = 'wav'
        tts = TTS(model_name=voice)
        if torch.cuda.is_available():
            tts.to('cuda:0')
    if model == 'tortoise':
        extension = 'wav'
        voice_samples, conditioning_latents = load_voices([voice])
        tts = TextToSpeech(half=half)

    # Syntetize audios
    start = pysrt.SubRipTime.from_ordinal(0)
    for number, sub in enumerate(subs):
        print(f" --> Processing subtitle {number}")
        duration = rest_subs(sub.start, start) # a duration
        save_silent(duration, f'{number}.1', model, extension)

        if model == 'gtts':
            t2v_gtts(sub.text, voice, number, extension)
        if model == 'coqui':
            t2v_coqui(tts, sub.text, number, extension)
        if model == 'tortoise':
            t2v_tortoise(tts, emotion, sub.text, voice_samples, custom_preset, number, extension)

        duration = rest_subs(sub.end, sub.start) # audio duration
        start = sub.end
        shrink_audio(duration, number, model, extension)
        torch.cuda.empty_cache()

    # Get the list of audio files in the directory and sort them
    files = [f for f in os.listdir("results/tmp/") if f.endswith(f".{extension}")]
    files = sorted(files, key=file_key)

    # Create a command for ffmpeg to join the files
    command = ["ffmpeg", "-loglevel", "quiet", "-y"]
    for f in files:
        command.extend(["-i", os.path.join("results/tmp/", f)])
    command.extend(["-filter_complex", "concat=n={}:v=0:a=1".format(len(files)), f"results/tmp/output.{extension}"])
    subprocess.run(command)

    # Final shrink
    command = f"ffprobe -i results/tmp/output.{extension} -show_entries format=duration -v quiet -of csv=\"p=0\" -loglevel quiet"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_p = process.communicate()[0].decode("utf-8")
    duration_file = float(output_p.strip())
    if abs(duration_file - subs[-1].end.ordinal / 1000) > 0.5:
        atempo = (duration_file) / (subs[-1].end.ordinal / 1000)
        print(f"Final atempo: {round(atempo, 4)}x")
        command = f"ffmpeg -i results/tmp/output.{extension} -filter:a \"atempo={atempo}\" -vn -loglevel quiet results/output.{extension}"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output_p = process.communicate()[0].decode("utf-8")
    else:
        shutil.move(f'results/tmp/output.{extension}', f'results/output.{extension}')

    shutil.rmtree("results/tmp/")
    return gr.Audio.update(f"results/output.{extension}", label=f"Total time {int(time.time() - start_time)}s")

##########################################################################

voices_gtts = {"Afrikaans": "af", "Arabic": "ar", "Bulgarian": "bg", "Bengali": "bn", "Bosnian": "bs", "Catalan": "ca", "Czech": "cs", "Danish": "da", "German": "de", "Greek": "el", "English": "en", "Spanish": "es", "Estonian": "et", "Finnish": "fi", "French": "fr", "Gujarati": "gu", "Hindi": "hi", "Croatian": "hr", "Hungarian": "hu", "Indonesian": "id", "Icelandic": "is", "Italian": "it", "Hebrew": "iw", "Japanese": "ja", "Javanese": "jw", "Khmer": "km", "Kannada": "kn", "Korean": "ko", "Latin": "la", "Latvian": "lv", "Malayalam": "ml", "Marathi": "mr", "Malay": "ms", "Myanmar (Burmese)": "my", "Nepali":"ne", "Dutch": "nl", "Norwegian": "no", "Polish": "pl", "Portuguese": "pt", "Romanian": "ro", "Russian": "ru", "Sinhala": "si", "Slovak": "sk", "Albanian": "sq", "Serbian": "sr", "Sundanese": "su", "Swedish": "sv", "Swahili": "sw", "Tamil": "ta", "Telugu": "te", "Thai": "th", "Filipino": "tl", "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur", "Vietnamese": "vi", "Chinese (Simplified)": "zh-CN", "Chinese (Mandarin/Taiwan)": "zh-TW", "Chinese (Mandarin)": "zh"}
voices_coqui = TTS().list_models()
voices_tortoise = os.listdir('tortoise/voices')
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
            model = gr.Dropdown(['gtts', 'coqui', 'tortoise'], value='tortoise', label='TTS model')
            subtitle = gr.File(type='binary', label='Subtitles file (.srt)')
            voice = gr.Dropdown(voices_tortoise, value=voices_tortoise[0], label='Voice')
            emotion = gr.Dropdown(emotion_list, value=emotion_list[0], label='Emotion')
            preview = gr.Audio(f"tortoise/voices/{voice.value}/" + random.choice(os.listdir(f"tortoise/voices/{voice.value}/")), label='Preview')
            refresh_btn = gr.Button('Refresh')
        with gr.Column() as params:
            num_autoregressive_samples = gr.Slider(minimum=4, maximum=512, value=4, label='Autoregressive samples', step=1)
            diffusion_iterations = gr.Slider(minimum=4, maximum=512, value=80, label='Diffusion iterations', step=1)
            temperature = gr.Slider(minimum=0.1, maximum=1, value=0.8, label='Temperature')
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

    model.change(model_change, inputs=[model], outputs=[voice, emotion, preview, refresh_btn, params])
    voice.change(voice_change, inputs=[model, voice], outputs=[preview])
    refresh_btn.click(refresh_btn_click, outputs=[voice])
    generate_btn.click(generate_btn_click, [subtitle, voice, emotion, model, num_autoregressive_samples, diffusion_iterations, temperature, length_penalty, repetition_penalty, top_p, cond_free, cond_free_k, diffusion_temperature, half], outputs=[output])
    app.launch(share=False, debug=True)
