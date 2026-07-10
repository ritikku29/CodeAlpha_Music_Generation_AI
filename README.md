# CodeAlpha Task 3 - Music Generation with AI

## Description
This project generates music using Artificial Intelligence using an LSTM neural network. It reads MIDI files, trains a model, and generates a new music file.

## Features
- Read MIDI files
- Train an LSTM model
- Generate new music
- Save output as MIDI

## Technologies Used
- Python
- TensorFlow
- Music21
- NumPy

## Project Structure

CodeAlpha_Music_Generation_AI/
│── train_music_model.py
│── generate_music.py
│── requirements.txt
│── README.md
│── dataset/
│── model/
└── output/

## Installation

```bash
pip install -r requirements.txt
```

## Run

Train the model:

```bash
python train_music_model.py
```

Generate music:

```bash
python generate_music.py
```

## Output

The generated music will be saved in:

```
output/generated_music.mid
```