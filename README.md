# Temporal Convolutional Networks

This code implements a temporal convolutional network, consisting on 1D convolutions and 1D poolings.

The data used for training and classification consists on raw inertial data, which is recommended to be normalized before
feeding the network.

## Prerequisites

To run this code, you will need Python3, tensorflow, numpy and sklearn.

To install the python libraries, use:

```
pip install tensorflow numpy sklearn
```

It is recomended to do this inside a virtual environment.

For a faster training, use tensorflow-gpu, here is a good guide to install it: https://github.com/williamFalcon/tensorflow-gpu-install-ubuntu-16.04


## Files

/data contains the datasets

/src contains the source code

## Running

To run the project, go to the src/ directory and type:

```
python main.py
```

This will train and evaluate the model.
