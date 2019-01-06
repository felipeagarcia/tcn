import numpy as np
from TCN import TCN
import data_handler as data
import os


if __name__ == '__main__':
    num_epochs = 1000
    n_classes = 18
    batch_size = 85
    num_features = 113
    timesteps = 24
    max_len = 150
    lean_rate = 0.001
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    inputs, labels, test_inputs, test_labels = data.load_data()
    model = TCN(n_classes)
    inputs = np.reshape(inputs, (-1, timesteps, num_features))
    inputs = np.transpose(inputs, (0, 2, 1))
    test_inputs = np.reshape(test_inputs, (-1, timesteps, num_features))
    test_inputs = np.transpose(test_inputs, (0, 2, 1))
    shape = inputs.shape[1:]
    model.create_tcn(shape)
    model.compile(lean_rate)
    model.fit(inputs, labels, num_epochs, batch_size)
    model.evaluate(test_inputs, test_labels, 300)

