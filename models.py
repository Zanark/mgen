from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import TimeDistributed
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