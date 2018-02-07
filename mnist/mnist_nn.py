from keras.models import Sequential
from keras.layers import Dense
import keras
import pandas as pd
import numpy as np
import os

def main():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print('in directory %s' % dir_path)

    dir = os.path.dirname(__file__)


    training_data = pd.read_csv(os.path.join(dir, './data/train.csv'), header=None)
    training_features = training_data.values[:, 1::]
    training_labels = training_data.values[:, 0]
    binary_training_labels = keras.utils.to_categorical(training_labels, num_classes=10)

    print('shape of training features = %s' % str(np.shape(training_features)))
    print('shape of training labels = %s' % str(np.shape(training_labels)))
    print('shape of binary training labels = %s' % str(np.shape(binary_training_labels)))

    test_data = pd.read_csv(os.path.join(dir, './data/test.csv'), header=None)
    test_features = test_data.values[:, 1::]
    test_labels = test_data.values[:, 0]
    binary_test_labels = keras.utils.to_categorical(test_labels, num_classes=10)

    print('shape of test features = %s' % str(np.shape(test_features)))
    print('shape of test labels = %s' % str(np.shape(test_labels)))
    print('shape of binary test labels = %s' % str(np.shape(binary_test_labels)))

    # building nn topology
    model = Sequential()
    model.add(Dense(units=300, activation='sigmoid', input_dim=784))
    model.add(Dense(units=10, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    training_epochs = 50

    model.fit(training_features, binary_training_labels, epochs=training_epochs, batch_size=100)
    test_accuracy = model.evaluate(test_features, binary_test_labels, batch_size=1000)[1]

    print('trained model accuracy on test set = %f' % test_accuracy)

if __name__ == '__main__':
    main()
