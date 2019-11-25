from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation

def nn_01(network_input, num_pitches):
    """ Super basic neural network for text generation taken from
        https://keras.io/examples/lstm_text_generation/ """
    model = Sequential()
    model.add(LSTM(128, input_shape=(network_input.shape[1], network_input.shape[2])))
    model.add(Dense(num_pitches))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model