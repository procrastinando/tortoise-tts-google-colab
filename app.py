import gradio as gr
import subprocess
import torchaudio
import torch
import pysrt
import shutil
import os
import random
import time
import yaml
from gtts import gTTS
from TTS.api import TTS

# If GPU:
models = ['gtts', 'coqui']
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

def save_silent(duration, name, model, extension):
    if model == 'gtts':
        command = ["ffmpeg", "-loglevel", "quiet", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(duration), "-acodec", "libmp3lame", f"results/tmp/{name}.{extension}"]
    if model == 'coqui':
        command = ["ffmpeg", "-loglevel", "quiet", "-y", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono", "-t", str(duration), "-acodec", "libmp3lame", f"results/tmp/{name}.{extension}"]
    if model == 'tortoise':
        command = ["ffmpeg", "-loglevel", "quiet", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=24000", "-t", str(duration), "-c:a", "pcm_f32le", f"results/tmp/{name}.{extension}"]
    subprocess.run(command, capture_output=True)
    print(f"Silence: {duration}s")

def audio_duration(file):
    command = f"ffprobe -i {file} -show_entries format=duration -v quiet -of csv=\"p=0\" -loglevel quiet"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_p = process.communicate()[0].decode("utf-8")
    return float(output_p.strip())

def shrink_audio(audio_file, atempo):
    filename, file_extension = os.path.splitext(audio_file)
    audio_file2 = f"{filename}2{file_extension}"
    command = ["ffmpeg", "-loglevel", "quiet", "-i", audio_file, "-filter:a", f"atempo={atempo}", audio_file2]
    subprocess.run(command, check=True)
    os.remove(audio_file)
    os.rename(audio_file2, audio_file)

def concatenate_audio(audio, audio2concatenate):
    command = ["ffmpeg", "-loglevel", "quiet", "-y", "-i", audio, "-i", audio2concatenate, "-filter_complex", "[0:0][1:0]concat=n=2:v=0:a=1[out]", "-map", "[out]", "results/tmp/final_output.wav"]
    subprocess.run(command)
    os.remove(audio)
    os.rename("results/tmp/final_output.wav", audio)

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
    return gr.Dropdown(choices=voices_clone, value=voices_clone[0])

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

def generate_btn_click(in_type, text, subtitle, voice, voice_clone, speech_lang, preview, max_atempo, min_atempo, audio_atempo, emotion, model, num_autoregressive_samples, diffusion_iterations, temperature, length_penalty, repetition_penalty, top_p, cond_free, cond_free_k, diffusion_temperature, half, compress, clean_files, progress=gr.Progress()):
    start_time = time.time()
    # Clean temporary files
    if clean_files:
        if os.path.exists("results/tmp/"):
            shutil.rmtree("results/tmp/")

    if model == 'gtts':
        extension = 'mp3'
    if model == 'coqui':
        extension = 'wav'
        if voice == voices_coqui[0] or voice == voices_coqui[1]:
            multilingual = True
        else:
            multilingual = False
        tts = TTS(model_name=voice)
        if torch.cuda.is_available():
            tts.to('cuda:0')
    if model == 'tortoise':
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
    if model == 'bark':
        extension = 'wav'

    # If text
    if in_type == 'Text':
        os.makedirs('results/', exist_ok=True)
        torch.cuda.empty_cache()
        log_start = time.time()
        if model == 'gtts':
            t2v_gtts(text, voice, f'results/speech.{extension}')
        if model == 'coqui':
            t2v_coqui(tts, text, f'results/speech.{extension}', multilingual, preview, speech_lang)
        if model == 'tortoise':
            t2v_tortoise(tts, emotion, text, voice_samples, custom_preset, f'results/speech.{extension}')
        if model == 'bark':
            t2v_bark(text, voice.split(' (')[0], f'results/speech.{extension}')
        output_log = f"* Synthesis time: {int(time.time() - log_start)}s"

        if audio_atempo != 1.0:
            shrink_audio(f"results/speech.{extension}", audio_atempo)

        if compress == 'No':
            out_audios = [f'results/speech.{extension}']
        else:
            command = ["ffmpeg", "-loglevel", "quiet", "-y", "-i", f'results/speech.{extension}', "-b:a", compress, f'results/speech_{compress}.mp3']
            subprocess.run(command, capture_output=True)
            os.remove(f'results/speech.{extension}')
            out_audios = [f'results/speech_{compress}.mp3']

        torch.cuda.empty_cache()

    # If subtitles
    elif in_type == 'Subtitles':
        out_audios = []
        output_log = ""
        for i in subtitle:
            try:
                os.makedirs('results/tmp/', exist_ok=True)
                subs = pysrt.open(i.name)
                save_silent(0, 'output', model, extension)
                output_file = f"results/tmp/output.{extension}"

                output_log = output_log + f"{os.path.basename(i.name)}:\n"
                log_start = time.time()
                log_silence = 0
                log_atempo = 0
                # Syntetize audios
                for number in progress.tqdm(range(len(subs)), desc=os.path.basename(i.name)):
                    torch.cuda.empty_cache()
                    print(f" --> Processing subtitle {number}")

                    if model == 'gtts':
                        t2v_gtts(subs[number].text, voice, f'results/tmp/{number}.speech.{extension}')
                    if model == 'coqui':
                        t2v_coqui(tts, subs[number].text, f'results/tmp/{number}.speech.{extension}', multilingual, preview, speech_lang)
                    if model == 'tortoise':
                        t2v_tortoise(tts, emotion, subs[number].text, voice_samples, custom_preset, f'results/tmp/{number}.speech.{extension}')
                    if model == 'bark':
                        t2v_bark(subs[number].text, voice.split(' (')[0], f'results/tmp/{number}.speech.{extension}')

                    if min_atempo != 1.0:
                        shrink_audio(f"results/tmp/{number}.speech.{extension}", min_atempo)

                    output_duration = audio_duration(f"results/tmp/output.{extension}")
                    start_sub = to_seconds(subs[number].start)
                    print(start_sub, 10*'-',output_duration)
                    if start_sub - output_duration >= 0:
                        save_silent(start_sub - output_duration, f'{number}.silent', model, extension)
                        concatenate_audio(output_file, f"results/tmp/{number}.silent.{extension}")
                        concatenate_audio(output_file, f"results/tmp/{number}.speech.{extension}")
                        log_silence = log_silence + (start_sub - output_duration)
                        log_atempo = log_atempo + min_atempo
                    else:
                        tts_duration = audio_duration(f"results/tmp/{number}.speech.{extension}")
                        if abs(start_sub - output_duration) > (tts_duration / (max_atempo/(max_atempo-1))):
                            atempo = max_atempo
                        else:
                            atempo = tts_duration / (tts_duration + (start_sub - output_duration))
                        print(f"Accelerate by {round(atempo, 4)}x")
                        shrink_audio(f"results/tmp/{number}.speech.{extension}", atempo)
                        concatenate_audio(output_file, f"results/tmp/{number}.speech.{extension}")
                        log_atempo = log_atempo + atempo

                    torch.cuda.empty_cache()

                output_log = output_log + f"* Synthesis time: {int(time.time() - log_start)}s\n"
                output_log = output_log + f"* Total silence: {int(log_silence)}s ({round(100*log_silence/audio_duration(f'results/tmp/output.{extension}'), 2)}%)\n"
                output_log = output_log + f"* Average atempo: {round(log_atempo/len(subs), 2)}x\n"

                shutil.move(f'results/tmp/output.{extension}', f'results/{os.path.basename(i.name)}.{extension}')
                if compress == 'No':
                    out_audios.append(f'results/{os.path.basename(i.name)}.{extension}')
                else:
                    command = ["ffmpeg", "-loglevel", "quiet", "-y", "-i", f'results/{os.path.basename(i.name)}.{extension}', "-b:a", compress, f'results/{os.path.basename(i.name)}_{compress}.mp3']
                    subprocess.run(command, capture_output=True)
                    os.remove(f'results/{os.path.basename(i.name)}.{extension}')
                    out_audios.append(f'results/{os.path.basename(i.name)}_{compress}.mp3')

                if clean_files:
                    shutil.rmtree("results/tmp/")
            except:
                print(f"{100*'-'}\nError opening this subtitle file: {i.name}\n{100*'-'}")

    return gr.File(out_audios, label=f"Total time {int(time.time() - start_time)}s"), gr.Textbox(output_log, visible=True)

##########################################################################

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
            text = gr.Textbox(label='Text', placeholder='Write something...')
            subtitle = gr.File(label='Subtitles file (.srt)', file_count='multiple', file_types=['.srt'], visible=False)
            model = gr.Dropdown(models, value='coqui', label='TTS model')
            voice = gr.Dropdown(choices=voices_coqui, value=voices_coqui[0], label='Voice model')
            with gr.Row():
                voice_clone = gr.Dropdown(choices=voices_clone, value=voices_clone[0], label='Voice to clone')
                refresh_btn = gr.Button('Refresh')
            speech_lang = gr.Dropdown(xtts_lang, value=xtts_lang[0], label='Speech language')
            preview = gr.Audio(value=f"tortoise/voices/{voice_clone.value}/" + random.choice(os.listdir(f"tortoise/voices/{voice_clone.value}/")), label='Preview', type='filepath')
            emotion = gr.Dropdown(emotion_list, value=emotion_list[0], label='Emotion', visible=False)
            with gr.Row():
                min_atempo = gr.Slider(0.5, 1.5, value=1.0, label="Minimum atempo", visible=False)
                max_atempo = gr.Slider(1, 2, value=1.2, label="Maximum atempo", visible=False)
            audio_atempo = gr.Slider(0.5, 2, value=1.0, label="Audio atempo")
        with gr.Column(visible=False) as params:
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
            compress = gr.Radio(choices=['No', '32k', '64k', '96k', '128k', '256k'], label='Compress audio', value='64k')
            with gr.Row():
                generate_btn = gr.Button('Generate')
                clean_files = gr.Checkbox(label='Clean temporary files', value=True)
            output = gr.File(label='Output', file_count='multiple')
            output_log = gr.Textbox(label='Log', visible=False)

    in_type.change(in_type_change, inputs=[in_type], outputs=[text, subtitle, max_atempo, min_atempo, audio_atempo])
    model.change(model_change, inputs=[model, voice], outputs=[voice, voice_clone, emotion, preview, refresh_btn, params, speech_lang])
    voice.change(voice_change, inputs=[model, voice], outputs=[voice_clone, refresh_btn, speech_lang, preview])
    voice_clone.change(voice_clone_change, inputs=[model, voice, voice_clone], outputs=[preview])
    refresh_btn.click(refresh_btn_click, outputs=[voice_clone])
    max_atempo.change(max_atempo_change, inputs=[max_atempo, min_atempo], outputs=[min_atempo])
    min_atempo.change(min_atempo_change, inputs=[min_atempo, max_atempo], outputs=[max_atempo])
    generate_btn.click(generate_btn_click, [in_type, text, subtitle, voice, voice_clone, speech_lang, preview, max_atempo, min_atempo, audio_atempo, emotion, model, num_autoregressive_samples, diffusion_iterations, temperature, length_penalty, repetition_penalty, top_p, cond_free, cond_free_k, diffusion_temperature, half, compress, clean_files], outputs=[output, output_log])
    app.queue().launch(share=False, debug=True)
