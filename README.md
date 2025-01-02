# CISM
# CISM (Chinese Instrument Separation Model)

Official implementation of CISM, a deep learning model specialized for separating traditional Chinese instruments using frequency band-based processing. The model focuses on instruments like guzheng, dizi, pipa, and xiao, utilizing frequency attention and multi-head attention mechanisms.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To perform music source separation:

```bash
cd CISM_main
chmod +x run_inference.sh
./run_inference.sh
```

Separated tracks will be saved in `test_music_output/`.
The model weights can be downloaded from xxxxxxx.com.
Demo ：https://huggingface.co/spaces/NMLAB8/CISM
