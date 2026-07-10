import os
import numpy as np
from tensorflow.keras.models import load_model
from music21 import stream, note

# Load trained model
model = load_model("model/music_model.keras")

# Dummy seed sequence
seed = np.random.rand(1, 50, 1)

generated_notes = []

for _ in range(100):
    prediction = model.predict(seed, verbose=0)
    index = np.argmax(prediction)

    generated_notes.append(index)

    seed = np.roll(seed, -1, axis=1)
    seed[0][-1] = index / prediction.shape[1]

# Convert generated notes to MIDI
output_notes = []

for pattern in generated_notes:
    new_note = note.Note(int(pattern % 60) + 30)
    new_note.offset = len(output_notes)
    output_notes.append(new_note)

midi_stream = stream.Stream(output_notes)

os.makedirs("output", exist_ok=True)
midi_stream.write("midi", fp="output/generated_music.mid")

print("Music generated successfully!")