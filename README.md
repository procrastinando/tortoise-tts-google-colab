# Google Colab Fork for Tortoise-tts

This repository has modified the file `tortoise/api.py` adding the function: `tts_custom` to allow use a gradio WebUI.

### 1. Installation:
Install python 3.9:
```
!wget -O mini.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
!chmod +x mini.sh
!bash ./mini.sh -b -f -p /usr/local
!update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python3.9 1
!python --version
!apt-get update
!apt install software-properties-common
!sudo dpkg --remove --force-remove-reinstreq python3-pip python3-setuptools python3-wheel
!apt-get install python3-pip
!pip install ipywidgets
```
Verify that a GPU is being used:
```
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
```
Install tortoise-tts:
```
!conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
!conda install transformers=4.29.2 -y
!git clone https://github.com/neonbjb/tortoise-tts.git
%cd /content/tortoise-tts
!python setup.py install
```
### 2. Run using `do_tts.py`:
Args:
* --text: the text to syntetize
* --voice: select a voice from the directory `tortoise/voices/`
* --preset: select a preset from [ultra_fast, fast, standard, high_quality]
* --output_path: select a path to export, by default: `results/`
* --candidates: How many candidates will be generated, by default: `3`
* --half: Half precition to reduce the ram and vram consumption, by default: `True`
```
!python /content/tortoise-tts/tortoise/do_tts.py --text "I'm going to speak this" --voice random --preset fast
```
### 3. Run the WebUI (does not work on colab):
Add this fuction to `tortoise/api.py`:
```
    def tts_custom(self, text, custom_preset, **kwargs):
        custom_preset.update(kwargs)
        return self.tts(text, **custom_preset)
```
Run the webUI:
```
python app.py
```
![image](https://github.com/procrastinando/tortoise-tts-google-colab/assets/74340724/eb0ad3e2-2904-445f-b42c-227d420d85bf)
