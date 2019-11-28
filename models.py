from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Bidirectional
from keras.layers import Dropout

def nn_01(network_input, num_pitches):
    """ Super basic neural network for text generation taken from
        https://keras.io/examples/lstm_text_generation/ """
    model = Sequential()
    model.add(LSTM(128, input_shape=(network_input.shape[1], network_input.shape[2])))
    model.add(Dense(num_pitches))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def nn_02(network_input, num_pitches):
    """ Double the number of LSTM cells as nn_01 """
    model = Sequential()
    model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2])))
    model.add(Dense(num_pitches))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def nn_03(network_input, num_pitches):
    """ 2 Stacked LSTMs """
    model = Sequential()
    model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(num_pitches))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def nn_04(network_input, num_pitches):
    """ 2 Stacked LSTMs with dropout in between """
    model = Sequential()
    model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(num_pitches))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def nn_04_1(network_input, num_pitches):
    """ 2 Stacked LSTMs with dropout at end """
    model = Sequential()
    model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(num_pitches))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def nn_05(network_input, num_pitches):
    """ 3 Stacked LSTM-256s with dropouts """
    model = Sequential()
    model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(256))
    model.add(Dense(num_pitches))
    model.add(Dropout(0.2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def nn_05_1(network_input, num_pitches):
    """ 3 Stacked LSTM-512s with dropouts """
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(512))
    model.add(Dense(num_pitches))
    model.add(Dropout(0.2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def nn_06(network_input, num_pitches):
    """ LSTM with BiDirectional LSTMs """
    model = Sequential()
    model.add(LSTM(
        256, 
        input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True)
    )
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dense(num_pitches))
    model.add(Dropout(0.2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model