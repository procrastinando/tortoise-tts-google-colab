import gradio as gr
import subprocess
import torchaudio
import torch
import pysrt
import shutil
import os
import glob
import random
import time
from datetime import datetime
import yaml
from gtts import gTTS
from TTS.api import TTS
models = ['gtts', 'coqui']

# If GPU:
if torch.cuda.is_available():
    models = models + ['tortoise', 'bark']
    # tortoise
    from tortoise.api import TextToSpeech
    from tortoise.utils.audio import load_voices
    # bark
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write
    import numpy as np
    import nltk
    nltk.download('punkt')

# Functions
def list_files(directory, extensions):
    files = []
    for extension in extensions:
        files.extend([os.path.basename(file) for file in glob.glob(f"{directory}/*{extension}")])
    return files

def list_dir(directory):
    return [f for f in os.listdir(directory) if not os.path.isfile(os.path.join(directory, f))]

# Media functions
def t2v_gtts(text, voice, audio_path):
    tts = gTTS(text, lang=voices_gtts[voice])
    tts.save(audio_path)

def t2v_coqui(tts, text, audio_path, multilingual, preview, lang):
    if multilingual:
        tts.tts_to_file(text=text, speaker_wav=preview, language=lang, file_path=audio_path)
    else:
        tts.tts_to_file(text=text, file_path=audio_path)

def t2v_tortoise(tts, emotion, text, voice_samples, custom_preset, audio_path):
    if type(emotion) is not str:
        gen = tts.tts_custom(text, voice_samples=voice_samples, custom_preset=custom_preset)
    else:
        gen = tts.tts_custom(emotion+text, voice_samples=voice_samples, custom_preset=custom_preset)
    torchaudio.save(audio_path, gen.squeeze(0).cpu(), 24000)

def t2v_bark(text, voice, audio_path):
    preload_models()
    script = text.replace("\n", " ").strip()
    sentences = nltk.sent_tokenize(script)
    silence = np.zeros(int(0.25 * SAMPLE_RATE)) # quarter second of silence
    pieces = []
    for sentence in sentences:
        audio_array = generate_audio(sentence, history_prompt=voice)
        pieces += [audio_array, silence.copy()]
    write(audio_path, SAMPLE_RATE, np.concatenate(pieces))

def save_silent(duration, file_path, model):
    print(model)
    if model == 'gtts':
        command = ["ffmpeg", "-loglevel", "quiet", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(duration), "-acodec", "libmp3lame", file_path]
    if model == 'coqui':
        command = ["ffmpeg", "-loglevel", "quiet", "-y", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono", "-t", str(duration), "-acodec", "libmp3lame", file_path]
    if model == 'tortoise':
        command = ["ffmpeg", "-loglevel", "quiet", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=24000", "-t", str(duration), "-c:a", "pcm_f32le", file_path]
    if model == 'bark':
        command = ["ffmpeg", "-loglevel", "quiet", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(duration), "-acodec", "pcm_f32le", file_path]
    subprocess.run(command, capture_output=True)
    print(f"Silence: {duration}s")

def audio_duration(file):
    command = f"ffprobe -i {file} -show_entries format=duration -v quiet -of csv=\"p=0\" -loglevel quiet"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_p = process.communicate()[0].decode("utf-8")
    return float(output_p.strip())

def concatenate_audio(audio, audio2concatenate):
    temp_file, extension = os.path.splitext(audio)
    temp_file = temp_file + "_temp" + extension
    command = ["ffmpeg", "-loglevel", "quiet", "-y", "-i", audio, "-i", audio2concatenate, "-filter_complex", "[0:0][1:0]concat=n=2:v=0:a=1[out]", "-map", "[out]", temp_file]
    subprocess.run(command)
    os.remove(audio)
    os.rename(temp_file, audio)

def to_seconds(srt_time):
    return (srt_time.hours * 60 * 60 +
            srt_time.minutes * 60 +
            srt_time.seconds +
            srt_time.milliseconds / 1000)

##########################################################################

def in_type_change(in_type):
    if in_type == 'Text':
        text = gr.Textbox(visible=True)
        subtitle = gr.File(visible=False)
        max_atempo = gr.Slider(visible=False)
        min_atempo = gr.Slider(visible=False)
        audio_atempo = gr.Slider(visible=True)
    elif in_type == 'Subtitles':
        text = gr.Textbox(visible=False)
        subtitle = gr.File(visible=True)
        max_atempo = gr.Slider(visible=True)
        min_atempo = gr.Slider(visible=True)
        audio_atempo = gr.Slider(visible=False)
    return text, subtitle, max_atempo, min_atempo, audio_atempo

def model_change(model, voice):
    if model == 'gtts':
        voice = gr.Dropdown(choices=list(voices_gtts.keys()), value=list(voices_gtts.keys())[10], visible=True)
        voice_clone = gr.Dropdown(visible=False)
        emotion = gr.Dropdown(visible=False)
        preview = gr.Audio(visible=False)
        refresh_btn = gr.Button(visible=False)
        params = gr.Column.update(visible=False)
        speech_lang = gr.Dropdown(visible=False)

    elif model == 'coqui':
        voice = gr.Dropdown(choices=voices_coqui, value=voices_coqui[0], visible=True)
        emotion = gr.Dropdown(visible=False)
        params = gr.Column.update(visible=False)
        voice_clone = gr.Dropdown(choices=voices_clone, value=voices_clone[0], visible=True)
        refresh_btn = gr.Button(visible=True)
        preview = gr.Audio(value=f"tortoise/voices/{voices_clone[0]}/" + random.choice(os.listdir(f"tortoise/voices/{voices_clone[0]}/")), visible=True)
        if voice == voices_coqui[0]:
            speech_lang = gr.Dropdown(xtts_lang, value=xtts_lang[0], visible=True)
        else:
            speech_lang = gr.Dropdown(yourtts_lang, value=yourtts_lang[0], visible=True)

    elif model == 'tortoise':
        voice = gr.Dropdown(visible=False)
        voice_clone = gr.Dropdown(choices=voices_clone, value=voices_clone[0], visible=True)
        emotion = gr.Dropdown(visible=True)
        preview = gr.Audio(value=f"tortoise/voices/{voices_clone[0]}/" + random.choice(os.listdir(f"tortoise/voices/{voices_clone[0]}/")), visible=True)
        refresh_btn = gr.Button(visible=True)
        params = gr.Column.update(visible=True)
        speech_lang = gr.Dropdown(visible=False)

    elif model == 'bark':
        voice = gr.Dropdown(choices=voices_bark, value=voices_bark[6], visible=True)
        voice_clone = gr.Dropdown(visible=False)
        emotion = gr.Dropdown(visible=False)
        preview = gr.Audio(value="https://dl.suno-models.io/bark/prompts/prompt_audio/en_speaker_6.mp3", visible=True)
        refresh_btn = gr.Button(visible=False)
        params = gr.Column.update(visible=False)
        speech_lang = gr.Dropdown(visible=False)

    return voice, voice_clone, emotion, preview, refresh_btn, params, speech_lang

def voice_change(model, voice):
    if voice == voices_coqui[0] or voice == voices_coqui[1]:
        voice_clone = gr.Dropdown(choices=voices_clone, value=voices_clone[0], visible=True)
        preview = gr.Audio(value=f"tortoise/voices/{voices_clone[0]}/" + random.choice(os.listdir(f"tortoise/voices/{voices_clone[0]}/")), visible=True)
        refresh_btn = gr.Button(visible=True)
        if voice == voices_coqui[0]:
            speech_lang = gr.Dropdown(xtts_lang, value=xtts_lang[0], visible=True)
        else:
            speech_lang = gr.Dropdown(yourtts_lang, value=yourtts_lang[0], visible=True)
    elif model == 'bark':
        voice_clone = gr.Dropdown(visible=False)
        preview = gr.Audio(value=f"https://dl.suno-models.io/bark/prompts/prompt_audio/{voice.split('/')[1].split(' (')[0]}.mp3", visible=True)
        refresh_btn = gr.Button(visible=False)
        speech_lang = gr.Dropdown(visible=False)
    else:
        voice_clone = gr.Dropdown(visible=False)
        preview = gr.Audio(visible=False)
        refresh_btn = gr.Button(visible=False)
        speech_lang = gr.Dropdown(visible=False)

    return voice_clone, refresh_btn, speech_lang, preview

def voice_clone_change(model, voice, voice_clone):
    if model == 'tortoise':
        return gr.Audio(f"tortoise/voices/{voice_clone}/" + random.choice(os.listdir(f"tortoise/voices/{voice_clone}/")), visible=True)
    elif model == 'coqui':
        if voice == voices_coqui[0] or voice == voices_coqui[1]:
            return gr.Audio(f"tortoise/voices/{voice_clone}/" + random.choice(os.listdir(f"tortoise/voices/{voice_clone}/")), visible=True)
        else:
            return gr.Audio(visible=False)
    else:
        return gr.Audio(visible=False)

def refresh_btn_click():
    global voices_clone
    voices_clone = os.listdir('tortoise/voices')
    return gr.Dropdown(choices=voices_clone, value=voices_clone[0]), gr.Dropdown(list_dir('results'), label='Project')

def max_atempo_change(max_atempo, min_atempo):
    if max_atempo < min_atempo:
        return gr.Slider(value=max_atempo)
    else:
        return gr.Slider()

def min_atempo_change(min_atempo, max_atempo):
    if min_atempo > max_atempo:
        return gr.Slider(value=min_atempo)
    else:
        return gr.Slider()

def projects_change(projects):
    if projects == None:
        projects_model = gr.Dropdown(visible=False)
        audios = gr.Dropdown(visible=False)
        subtitles = gr.Dropdown(visible=False)
        subtitle_project = gr.Row.update(visible=False)
        text_project = gr.Row.update(visible=False)
        sync_btn = gr.Button('Run syncronization', visible=False)
    elif projects == 'text':
        projects_model = gr.Dropdown([dir for dir in list_dir(f'results/{projects}') if not dir.endswith('_sync')], label='Project model', visible=True)
        audios = gr.Dropdown(visible=False)
        subtitles = gr.Dropdown(visible=False)
        subtitle_project = gr.Row.update(visible=False)
        text_project = gr.Row.update(visible=True)
        sync_btn = gr.Button('Run settings', visible=True)
    else:
        projects_model = gr.Dropdown([dir for dir in list_dir(f'results/{projects}') if not dir.endswith('_sync')], label='Project model', visible=True)
        audios = gr.Dropdown(visible=False)
        subtitles = gr.Dropdown(list_files(f'results/{projects}', ['.srt']), label='Subtitle file', visible=True)
        subtitle_project = gr.Row.update(visible=True)
        text_project = gr.Row.update(visible=False)
        sync_btn = gr.Button('Run syncronization', visible=True)

    return projects_model, subtitles, subtitle_project, text_project, sync_btn

def output_list_change(projects, projects_model, output_list):
    return gr.Audio(value=f'results/{projects}/{projects_model}/{output_list}')

def generate_btn_click(in_type, text, subtitle, voice, voice_clone, speech_lang, preview, emotion, model, num_autoregressive_samples, diffusion_iterations, temperature, length_penalty, repetition_penalty, top_p, cond_free, cond_free_k, diffusion_temperature, half, progress=gr.Progress()):
    if model == 'gtts':
        extension = 'mp3'
    elif model == 'coqui':
        extension = 'wav'
        if voice == voices_coqui[0] or voice == voices_coqui[1]:
            multilingual = True
        else:
            multilingual = False
        tts = TTS(model_name=voice)
        if torch.cuda.is_available():
            tts.to('cuda:0')
    elif model == 'tortoise':
        extension = 'wav'
        voice_samples, conditioning_latents = load_voices([voice_clone])
        tts = TextToSpeech(half=half)
        # Parameters for tortoise
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
    elif model == 'bark':
        extension = 'wav'

    # If text
    if in_type == 'Text':
        time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(f'results/text/{time_stamp}-{model}', exist_ok=True)
        out_audios = []
        log_start = time.time()
        output_log = ""

        paragraph = text.split('\n')
        for number in progress.tqdm(range(len(paragraph))):
            torch.cuda.empty_cache()
            speech = paragraph[number].translate(str.maketrans('', '', '\\/*?:\"<>|'))

            if model == 'gtts':
                t2v_gtts(paragraph[number], voice, f'results/text/{time_stamp}-{model}/{str(number).zfill(4)}-{speech}.{extension}')
            if model == 'coqui':
                t2v_coqui(tts, paragraph[number], f'results/text/{time_stamp}-{model}/{str(number).zfill(4)}-{speech}.{extension}', multilingual, preview, speech_lang)
            if model == 'tortoise':
                t2v_tortoise(tts, emotion, paragraph[number], voice_samples, custom_preset, f'results/text/{time_stamp}-{model}/{str(number).zfill(4)}-{speech}.{extension}')
            if model == 'bark':
                t2v_bark(paragraph[number], voice.split(' (')[0], f'results/text/{time_stamp}-{model}/{str(number).zfill(4)}-{speech}.{extension}')

        output_log = f"* Synthesis time: {int(time.time() - log_start)}s"
        torch.cuda.empty_cache()

    # If subtitles
    elif in_type == 'Subtitles':
        output_log = ""
        for i in subtitle:
            project = os.path.splitext(os.path.basename(i.name))[0]
            os.makedirs(f'results/{project}/{model}/', exist_ok=True)
            shutil.copy(i.name, f'results/{project}/{project}.srt')
            subs = pysrt.open(f'results/{project}/{project}.srt')

            output_log = output_log + f"{os.path.basename(i.name)}:\n"
            log_start = time.time()
            # Syntetize audios
            for number in progress.tqdm(range(len(subs)), desc=project):
                torch.cuda.empty_cache()
                print(f" --> Processing subtitle {number}")

                if model == 'gtts':
                    t2v_gtts(subs[number].text, voice, f'results/{project}/{model}/{project}.{number}.{extension}')
                if model == 'coqui':
                    t2v_coqui(tts, subs[number].text, f'results/{project}/{model}/{project}.{number}.{extension}', multilingual, preview, speech_lang)
                if model == 'tortoise':
                    t2v_tortoise(tts, emotion, subs[number].text, voice_samples, custom_preset, f'results/{project}/{model}/{project}.{number}.{extension}')
                if model == 'bark':
                    t2v_bark(subs[number].text, voice.split(' (')[0], f'results/{project}/{model}/{project}.{number}.{extension}')
                torch.cuda.empty_cache()
            output_log = output_log + f"* Synthesis time: {int(time.time() - log_start)}s\n"

    print(output_log)
    return gr.Textbox(output_log, visible=True), gr.Dropdown(list_dir('results'), label='Project')

def sync_btn_click(projects, projects_model, subtitles, min_atempo, max_atempo, delta_atempo, audio_atempo, compress, progress=gr.Progress()):
    if projects == None or projects_model == None:
        output_list = gr.Dropdown(visible=False)
        output_audio = gr.Audio(visible=False)
        output_sync = gr.Audio(visible=False)
        sync_log = gr.Textbox(visible=False)

    elif projects == 'text':
        if projects_model == None:
            output_list = gr.Dropdown(visible=False)
            output_audio = gr.Audio(visible=False)
            output_sync = gr.Audio(visible=False)
            sync_log = gr.Textbox(visible=False)
        else:
            project_model = projects_model.split('-')[-1]
            if project_model == 'gtts':
                extension = 'mp3'
            else:
                extension = 'wav'
            sync_file = f'results/{projects}/{projects_model}_sync.{extension}'
            save_silent(0, sync_file, project_model)

            output_list = []
            output_audio = []
            speech_list = list_files(f'results/{projects}/{projects_model}', ['.'+extension])
            for i in speech_list:
                if os.path.exists(f'results/{projects}/{projects_model}_sync'):
                    shutil.rmtree(f'results/{projects}/{projects_model}_sync')
                os.makedirs(f'results/{projects}/{projects_model}_sync')
                input_path = f'results/{projects}/{projects_model}/{i}'
                output_path = f'results/{projects}/{projects_model}_sync/{i}'

                if audio_atempo == 1.0:
                    shutil.copy(input_path, output_path)
                else:
                    command = ["ffmpeg", "-loglevel", "quiet", "-i", input_path, "-filter:a", f"atempo={audio_atempo}", output_path]
                    subprocess.run(command, check=True)
                concatenate_audio(sync_file, output_path)

            if compress != "Don't":
                sync_file_compressed = f'results/{projects}/{projects_model}_sync_compressed.mp3'
                command = ["ffmpeg", "-loglevel", "quiet", "-y", "-i", sync_file, "-b:a", compress, sync_file_compressed]
                subprocess.run(command, capture_output=True)
                sync_file = sync_file_compressed

            output_list = gr.Dropdown(value=speech_list[0], choices=speech_list, visible=True)
            output_audio = gr.Audio(value=f'results/{projects}/{projects_model}/{speech_list[0]}', visible=True)
            output_sync = gr.Audio(sync_file, visible=True)
            sync_log = gr.Textbox(visible=False)

    else:
        if os.path.exists(f'results/{projects}/{projects_model}_sync'):
            shutil.rmtree(f'results/{projects}/{projects_model}_sync')
        os.makedirs(f'results/{projects}/{projects_model}_sync')

        if projects_model == 'gtts':
            extension = 'mp3'
        else:
            extension = 'wav'

        output_log = f"{subtitles}:\n"
        log_silence = 0
        log_overlap = 0
        log_atempo = 0
        log_duration = 0
        output_list = []
        output_audio = []

        atempo = [min_atempo, min_atempo, min_atempo, min_atempo*(1+delta_atempo), max_atempo]
        sync_file = f'results/{projects}/{projects}_sync.{extension}' # results/proyecto/proyecto_sync.wav
        save_silent(0, sync_file, projects_model)
        subs = pysrt.open(f'results/{projects}/{subtitles}')

        for number in progress.tqdm(range(len(subs)), desc=os.path.basename(subtitles)):
            speech_file = f'results/{projects}/{projects_model}/{projects}.{number}.{extension}'
            speech_file_sync = f'results/{projects}/{projects_model}_sync/{projects}.{number}.speech.{extension}'
            silent_file = f'results/{projects}/{projects_model}_sync/{projects}.{number}.silent.{extension}'

            speech_file_duration = audio_duration(speech_file)
            output_duration = audio_duration(sync_file)
            start_sub = to_seconds(subs[number].start)
            print(start_sub, 10*'-', output_duration)

            if start_sub-output_duration >= 0: # copy the file from the non sync directory and create a silence audio
                atempo[2] = max([atempo[0], atempo[1]])
                atempo[1] = atempo[2] / (1+delta_atempo)
                atempo[3] = atempo[2] * (1+delta_atempo)
                print(f"Speech speed {round(atempo[2], 4)}x")
                command = ["ffmpeg", "-loglevel", "quiet", "-i", speech_file, "-filter:a", f"atempo={atempo[2]}", speech_file_sync]
                subprocess.run(command, check=True)

                save_silent(start_sub-output_duration, silent_file, projects_model)
                concatenate_audio(sync_file, silent_file)
                concatenate_audio(sync_file, speech_file_sync)
                log_silence = log_silence + (start_sub - output_duration)

            else:
                atempo_up = min([atempo[3], atempo[4]])
                if (output_duration-start_sub) > (speech_file_duration / (atempo_up/(atempo_up-1))):
                    atempo_use = atempo_up
                else:
                    atempo_use = speech_file_duration / (speech_file_duration + (start_sub - output_duration))
                    atempo_use = max([atempo_use, atempo[0], atempo[1]])
                print(f"Speech speed {round(atempo_use, 4)}x")

                atempo[2] = atempo_use
                atempo[1] = atempo_use / (1+delta_atempo)
                atempo[3] = atempo_use * (1+delta_atempo)
                command = ["ffmpeg", "-loglevel", "quiet", "-i", speech_file, "-filter:a", f"atempo={atempo[2]}", speech_file_sync]
                subprocess.run(command, check=True)
                concatenate_audio(sync_file, speech_file_sync)
                log_overlap = log_overlap + (output_duration-start_sub)

            log_atempo = log_atempo + (atempo[2] * speech_file_duration)
            log_duration = log_duration + speech_file_duration

        output_log = output_log + f"* Total silence: {int(log_silence)}s ({round(100*log_silence/log_duration, 2)}%)\n"
        output_log = output_log + f"* Total overlap: {int(log_overlap)}s ({round(100*log_overlap/log_duration, 2)}%)\n"
        output_log = output_log + f"* Average atempo: {round(log_atempo/log_duration, 2)}x\n"
        output_log = output_log + "* Recommended settings:\n"
        output_log = output_log + f"  - Minimum atempo: {round((log_atempo/log_duration + log_overlap/log_duration - log_silence/log_duration) / 1.1, 2)}\n"
        output_log = output_log + f"  - Maximum atempo: {round((log_atempo/log_duration + log_overlap/log_duration - log_silence/log_duration) * 1.1, 2)}"

        if compress != "Don't":
            sync_file_compressed = f'results/{projects}/{projects}_sync_compressed.mp3'
            command = ["ffmpeg", "-loglevel", "quiet", "-y", "-i", sync_file, "-b:a", compress, sync_file_compressed]
            subprocess.run(command, capture_output=True)
            sync_file = sync_file_compressed

        speech_list = list_files(f'results/{projects}/{projects_model}', ['.'+extension])
        output_list = gr.Dropdown(value=speech_list[0], choices=speech_list, visible=True)
        output_audio = gr.Audio(value=f'results/{projects}/{projects_model}/{speech_list[0]}', visible=True)
        output_sync = gr.Audio(sync_file, visible=True)
        sync_log = gr.Textbox(output_log, visible=True)

    print(sync_log)
    return output_list, output_audio, output_sync, sync_log

##########################################################################

if not os.path.exists("results/"):
    os.makedirs('results/')

with open('data.yaml', 'r') as file:
    data = yaml.safe_load(file)

voices_gtts = data['gtts']
voices_coqui = data['coqui']
voices_bark = data['bark']
voices_clone = os.listdir('tortoise/voices')
xtts_lang = data['xtts']
yourtts_lang = data['yourtts']
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
            in_type = gr.Radio(choices=['Text', 'Subtitles'], label='Input type', value='Text')
            text = gr.TextArea(show_label=False, placeholder='Write something...')
            subtitle = gr.File(label='Subtitles file (.srt)', file_count='multiple', file_types=['.srt'], visible=False)
            model = gr.Dropdown(models, value='coqui', label='TTS model')
            voice = gr.Dropdown(choices=voices_coqui, value=voices_coqui[0], label='Voice model')
            with gr.Row():
                voice_clone = gr.Dropdown(choices=voices_clone, value=voices_clone[0], label='Voice to clone')
                refresh_btn = gr.Button('Refresh')
            speech_lang = gr.Dropdown(xtts_lang, value=xtts_lang[0], label='Speech language')
            preview = gr.Audio(value=f"tortoise/voices/{voice_clone.value}/" + random.choice(os.listdir(f"tortoise/voices/{voice_clone.value}/")), label='Preview', type='filepath')
            emotion = gr.Dropdown(emotion_list, value=emotion_list[0], label='Emotion', visible=False)
        with gr.Column(visible=False) as params:
            gr.HTML("Tortoise settings")
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
            generate_btn = gr.Button('Generate speech')
            generate_log = gr.Textbox(show_label=False, visible=False)

            gr.HTML("Speech syncronization settings")
            with gr.Row():
                projects = gr.Dropdown(list_dir('results'), label='Project')
                projects_model = gr.Dropdown(visible=False)
                subtitles = gr.Dropdown(visible=False)
            with gr.Row(visible=False) as subtitle_project:
                min_atempo = gr.Slider(0.5, 1.5, value=1.0, label="Minimum atempo")
                max_atempo = gr.Slider(1, 2, value=1.2, label="Maximum atempo")
                delta_atempo = gr.Slider(0.01, 1, value=0.1, label="Maximum atempo variation")
            with gr.Row(visible=False) as text_project:
                audio_atempo = gr.Slider(0.5, 2, value=1.0, label="Audio atempo")
            compress = gr.Radio(choices=["Don't", '32k', '64k', '128k', '256k'], label='Compress audio', value='64k')

            sync_btn = gr.Button('Run syncronization', visible=False)
            with gr.Row():
                output_list = gr.Dropdown(label="Original audio", visible=False)
                output_audio = gr.Audio(show_label=False, visible=False)
            output_sync = gr.Audio(label="Concatenated audio", visible=False)
            sync_log = gr.Textbox(show_label=False)

    in_type.change(in_type_change, inputs=[in_type], outputs=[text, subtitle, max_atempo, min_atempo, audio_atempo])
    model.change(model_change, inputs=[model, voice], outputs=[voice, voice_clone, emotion, preview, refresh_btn, params, speech_lang])
    voice.change(voice_change, inputs=[model, voice], outputs=[voice_clone, refresh_btn, speech_lang, preview])
    voice_clone.change(voice_clone_change, inputs=[model, voice, voice_clone], outputs=[preview])
    refresh_btn.click(refresh_btn_click, outputs=[voice_clone, projects])
    generate_btn.click(generate_btn_click, [in_type, text, subtitle, voice, voice_clone, speech_lang, preview, emotion, model, num_autoregressive_samples, diffusion_iterations, temperature, length_penalty, repetition_penalty, top_p, cond_free, cond_free_k, diffusion_temperature, half], outputs=[generate_log, projects])
    projects.change(projects_change, inputs=[projects], outputs=[projects_model, subtitles, subtitle_project, text_project, sync_btn])
    max_atempo.change(max_atempo_change, inputs=[max_atempo, min_atempo], outputs=[min_atempo])
    min_atempo.change(min_atempo_change, inputs=[min_atempo, max_atempo], outputs=[max_atempo])
    sync_btn.click(sync_btn_click, [projects, projects_model, subtitles, min_atempo, max_atempo, delta_atempo, audio_atempo, compress], outputs=[output_list, output_audio, output_sync, sync_log])
    output_list.change(output_list_change, inputs=[projects, projects_model, output_list], outputs=[output_audio])

    app.queue().launch(share=True, debug=True)
