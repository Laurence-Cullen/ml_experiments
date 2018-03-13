from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
import os


def main():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print('in directory %s' % dir_path)

    working_directory = os.path.dirname(__file__)

    training_data = pd.read_csv(os.path.join(working_directory, './data/train.csv'), header=None)
    training_features = training_data.values[:, 1::]
    training_features = (training_features - np.nanmean(training_features)) / np.nanstd(training_features)
    training_labels = training_data.values[:, 0]
    binary_training_labels = keras.utils.to_categorical(training_labels, num_classes=10)

    print('shape of training features = %s' % str(np.shape(training_features)))
    print('shape of training labels = %s' % str(np.shape(training_labels)))
    print('shape of binary training labels = %s' % str(np.shape(binary_training_labels)))

    test_data = pd.read_csv(os.path.join(working_directory, './data/test.csv'), header=None)
    test_features = test_data.values[:, 1::]
    test_features = (test_features - np.nanmean(test_features)) / np.nanstd(test_features)
    test_labels = test_data.values[:, 0]
    binary_test_labels = keras.utils.to_categorical(test_labels, num_classes=10)

    print('shape of test features = %s' % str(np.shape(test_features)))
    print('shape of test labels = %s' % str(np.shape(test_labels)))
    print('shape of binary test labels = %s' % str(np.shape(binary_test_labels)))

    dropout = 0.02

    # building nn topology
    model = Sequential()
    model.add(Dropout(dropout, input_shape=(784,)))
    model.add(Dense(units=784, activation='relu', input_dim=784))
    model.add(Dropout(dropout))
    model.add(Dense(units=400, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=300, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=200, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=150, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=10, activation='sigmoid'))

    optimizer = optimizers.sgd(lr=0.1, momentum=0.9, decay=0.2)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    training_epochs = 50
    #
    # model.fit(training_features, binary_training_labels, epochs=training_epochs, batch_size=100)
    # test_accuracy = model.evaluate(test_features, binary_test_labels, batch_size=1000)[1]

    # hold historical training and test accuracy
    train_accuracy = {}
    test_accuracy = {}

    epoch = 0

    try:
        while epoch < training_epochs:
            model.fit(training_features, binary_training_labels, epochs=1, batch_size=128)
            test_accuracy[epoch] = model.evaluate(test_features, binary_test_labels, batch_size=128)[1]
            train_accuracy[epoch] = model.evaluate(training_features, binary_training_labels, batch_size=128)[1]

            # TODO add sequential model saving

            print('\nepoch = %i\n' % epoch)

            epoch += 1
    except KeyboardInterrupt:
        pass

    # plotting training and test accuracy histories
    plt.plot(train_accuracy.keys(), train_accuracy.values(), label='train')
    plt.plot(test_accuracy.keys(), test_accuracy.values(), label='test')
    axes = plt.gca()
    axes.set_ylim([0.9, 1.0])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    # print('trained model accuracy on test set = %f' % test_accuracy)


if __name__ == '__main__':
    main()
