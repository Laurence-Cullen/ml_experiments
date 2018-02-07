from __future__ import division
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense
import pandas as pd
import numpy as np

class ModelEvaluator(object):
    def __init__(self, true, predictions):
        self._true_positives = self.calculate_true_positives(true, predictions)
        self._true_negatives = self.calculate_true_negatives(true, predictions)
        self._false_positives = self.calculate_false_positives(true, predictions)
        self._false_negatives = self.calculate_false_negatives(true, predictions)


    def calculate_true_positives(self, true, predictions):
        return np.sum(np.logical_and(true, predictions))

    def calculate_false_positives(self, true, predictions):
        return np.sum(np.greater(predictions, true))

    def calculate_true_negatives(self, true, prediction):
        return np.sum(np.logical_not(np.logical_and(true, prediction)))

    def calculate_false_negatives(self, true, predictions):
        return np.sum(np.greater(true, predictions))

    @property
    def true_positives(self):
        return self._true_positives

    @property
    def true_negatives(self):
        return self._true_negatives

    @property
    def false_positives(self):
        return self._false_positives

    @property
    def false_negatives(self):
        return self._false_negatives

    @property
    def precision(self):
        return self._true_positives / (self._true_positives + self._false_positives)

    @property
    def recall(self):
        return true_positives(true, predictions) / (true_positives(true, predictions) + false_negatives(true, predictions))



def main():
    train_data = pd.read_csv('./data/train.csv')
    train_labels = np.greater(train_data['LABEL'].values,
                              np.ones(shape=np.shape(train_data['LABEL'].values), dtype=int)).astype(int)
    train_features = construct_features(train_data)
    
    test_data = pd.read_csv('./data/test.csv')
    test_labels = np.greater(test_data['LABEL'].values,
                              np.ones(shape=np.shape(test_data['LABEL'].values), dtype=int)).astype(int)
    test_features = construct_features(test_data)

    _, number_of_features = np.shape(train_features)

    # building nn topology
    # TODO experiment with adding regularization
    model = Sequential()
    model.add(Dense(units=2000, activation='relu', input_dim=number_of_features))
    model.add(Dense(units=30, activation='relu'))
    model.add(Dense(units=1, activation='relu'))

    optimizer = optimizers.sgd(lr=0.01)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['accuracy',])

    model.fit(train_features, train_labels, epochs=1000, batch_size=128)
    
    train_predictions = model.predict(train_features, batch_size=128)
    
    comparison_array = (0.5 * np.ones(shape=np.shape(train_predictions))).astype(dtype=np.float32)
    binary_train_predictions = np.greater(train_predictions.astype(dtype=np.float32), comparison_array)
    

def construct_features(kepler_data):
    local_kepler_data = kepler_data.copy()

    del local_kepler_data['LABEL']

    unscaled_kepler_features = local_kepler_data.values
    # print(unscaled_kepler_features)
    # print(np.shape(unscaled_kepler_features))

    rows, cols = np.shape(unscaled_kepler_features)

    scaled_kepler_features = np.zeros(shape=(rows, cols), dtype=float)

    for row in range(0, rows):
        # max_intensity = np.max(unscaled_kepler_features[row, :])
        # min_intensity = np.min(unscaled_kepler_features[row, :])
        # scaled_kepler_features[row, :] = 0.5 + (unscaled_kepler_features[row, :] /
        #                                         (2 * np.max(np.absolute(unscaled_kepler_features[row, :]))))
         scaled_kepler_features[row, :] = (unscaled_kepler_features[row, :] /
                                            np.max(np.absolute(unscaled_kepler_features[row, :])))

    return scaled_kepler_features

if __name__ == '__main__':
    main()
