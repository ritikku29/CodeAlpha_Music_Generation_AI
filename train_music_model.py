import os
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Path to dataset
DATASET_PATH = "dataset"

notes = []

# Read all MIDI files
for file in os.listdir(DATASET_PATH):
    if file.endswith(".mid") or file.endswith(".midi"):
        midi = converter.parse(os.path.join(DATASET_PATH, file))

        print(f"Reading {file}")

        parts = instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))

# Unique notes
pitchnames = sorted(set(notes))

note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

sequence_length = 50

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]

    network_input.append([note_to_int[n] for n in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(len(pitchnames))

network_output = to_categorical(network_output)

# Build LSTM model
model = Sequential()

model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(256))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(network_output.shape[1], activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam")

print("Training started...")

model.fit(network_input, network_output, epochs=20, batch_size=64)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/music_model.keras")

print("Model saved successfully!")