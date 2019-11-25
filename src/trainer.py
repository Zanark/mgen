""" This module prepares midi file data and feeds it to the neural
    network for training """
import numpy
from .notes import Notes
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

class Trainer:
    def __init__(self, path_to_data, path_to_notes, path_to_weights, length):
        self.notes = Notes(path_to_data, path_to_notes)
        self.sequence_length = length
        self.path_to_weights = path_to_weights

    def prepare(self):
        """ Prepares for training a model """
        self.notes.update_notes()
        notes = self.notes.notes

        self.num_pitches = len(set(notes))
        self.pitch_names = sorted(set(item for item in notes))
        return self.prepare_io(notes, self.num_pitches)
    
    def prepare_io(self, notes, num_pitches):
        """ Prepares the inputs and outputs used by the Neural Network """
        # map pitches to integers using a dictionary
        note_to_int = dict((note, number) for number, note in enumerate(self.pitch_names))

        nn_input = []
        nn_output = []

        # prepare inputs and the corresponding outputs
        for i in range(0, len(notes) - self.sequence_length, 1):
            sequence_in = notes[i:i + self.sequence_length]
            sequence_out = notes[i + self.sequence_length]
            nn_input.append([note_to_int[char] for char in sequence_in])
            nn_output.append(note_to_int[sequence_out])

        # make the outputs categorical (to one-hot vectors)
        nn_output = np_utils.to_categorical(nn_output)

        return (nn_input, nn_output)
    
    def get_normalized_input(self, nn_input):
        n_patterns = len(nn_input)
        # reshape the input for compatibility with LSTM layers
        normalized_input = numpy.reshape(nn_input, (n_patterns, self.sequence_length, 1))
        # normalize the input
        normalized_input = normalized_input / float(self.num_pitches)

        return normalized_input

    def train_model(self, model, nn_input, nn_output, batch_size, num_epochs):
        """ train the neural network """
        checkpoint = ModelCheckpoint(
            self.path_to_weights,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]

        model.fit(nn_input, nn_output, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks_list)
