import numpy as np
from TCN import TCN
import data_handler as data
import os


if __name__ == '__main__':
    num_epochs = 1000
    n_classes = 20
    batch_size = 20
    num_features = 19
    timesteps = 150
    max_len = 150
    lean_rate = 0.001
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    inputs, labels = data.open_data(max_len=max_len)
    inputs, labels = np.array(inputs), np.array(labels)
    model = TCN(n_classes)
    inputs = np.reshape(inputs, (-1, timesteps, num_features))
    shape = inputs.shape[1:]
    model.create_tcn(shape)
    model.compile(lean_rate)
    model.fit(inputs, labels, num_epochs, batch_size)
