# Automatic Speech Recognition using DeepSpeech2 model

## This repository contains implementation of DeepSpeech2 model for automatic speech recognition based on paper [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)

## Installation guide

Follow these steps to install the project:

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. To check metrics with pretrained model download best_model using this [link](https://drive.google.com/file/d/1_Imi79f23T28zqp-HoKuJ_yAM50gwgML/view?usp=sharing) config for this model can be found in `src/configs/deepspeech2_baseline.yaml`

## How to run training

To train a model, run the following command:

```bash
python3 train.py
```

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py
```
