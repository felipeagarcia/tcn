import tensorflow as tf


class TCN():
    def __init__(self, num_classes, kernel_size=(5),
                 filter_size=[128, 256, 256], pool_size=(2),
                 num_cnn_layers=3, dropout_rate=0.4):
        self.num_classes = num_classes
        self.pool_size = pool_size
        self.num_cnn_layers = num_cnn_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        if(len(filter_size) != num_cnn_layers):
            raise Exception("Filter size must be of the same lenght as\
                             num_cnn_layers")
        self.filter_size = filter_size
        self.model = None

    def create_tcn(self, input_shape):
        # input_shape must be (time_steps, num_features)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(self.filter_size[0], self.kernel_size,
                                         input_shape=input_shape))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPool1D(self.pool_size))
        for layer in range(1, self.num_cnn_layers):
            model.add(tf.keras.layers.Conv1D(self.filter_size[layer],
                      self.kernel_size))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPool1D(self.pool_size))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(self.num_classes,
                                        activation=tf.nn.softmax))
        self.model = model

    def compile(self, learn_rate):
        optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, input_data, labels, num_epochs, batch_size):
        self.model.fit(input_data, labels, epochs=num_epochs,
                       batch_size=batch_size, validation_split=0.2)

    def evaluate(self, test_data, test_labels, steps):
        loss, accuracy = self.model.evaluate(test_data, test_labels,
                                             steps=steps)
        print("Model loss:", loss, ", Accuracy:", accuracy)
        return loss, accuracy
