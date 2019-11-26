from src.trainer import Trainer
from src.prediction import Prediction
import models
import time

# Variables
get_nn = models.nn_01
fname = "01-01"
sequence_length = 100
generated_length = 500
num_epochs = 100
batch_size = 32

# Constants
path_to_notes = "./generated/data/notes"
path_to_weights = "./generated/data/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
path_to_data = "./dataset/*.mid"
path_to_midi = "./generated/%s.mid" % fname
path_to_log = "./generated/%s.log" % fname

if __name__ == '__main__':
    before_training = time.time()

    trainer = Trainer(path_to_data, path_to_notes, path_to_weights, sequence_length)
    nn_input, nn_output = trainer.prepare()
    normalized_input = trainer.get_normalized_input(nn_input)

    model = get_nn(normalized_input, trainer.num_pitches)
    trainer.train_model(model, normalized_input, nn_output, batch_size, num_epochs)

    after_training = time.time()

    # Load the weights to each node (if the prediction is being done later)
    # model.load_weights(path_to_weights)
    prediction = Prediction(trainer.notes.array, trainer.pitch_names, trainer.num_pitches)
    prediction_output = prediction.generate_notes(
        model, nn_input, generated_length
    )
    prediction.generate_midi(prediction_output, path_to_midi)

    after_prediction = time.time()

    with open(path_to_log, "w") as f:
        f.write("Training: ", after_training - before_training)
        f.write("Prediction: ", after_prediction - after_training)