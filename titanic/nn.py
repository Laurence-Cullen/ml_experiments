from __future__ import division
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# setting constant seed
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


def clean_ticket_numbers(ticket_strings):
    ticket_numbers = np.zeros(shape=np.shape(ticket_strings), dtype=int)

    for i in range(0, len(ticket_strings)):
        try:
            ticket_numbers[i] = ticket_strings[i].split()[-1]
        except TypeError:
            pass
        except ValueError:
            pass
        except IndexError:
            pass

    return ticket_numbers


def construct_features(passenger_data):
    number_of_features = 9
    examples = len(passenger_data['Age'].values)

    # creating features array
    features = np.zeros(shape=(examples, number_of_features))

    features[:, 0] = passenger_data['Age'].values
    sex_array = passenger_data['Sex'].values

    mean_age = np.nanmean(features[:, 0])
    max_age = np.nanmax(features[:, 0])

    # adding scaled passenger fare
    mean_fare = np.nanmean(passenger_data['Fare'].values)
    features[:, 3] = passenger_data['Fare'].values / np.nanmax(passenger_data['Fare'].values)

    for i in range(0, examples):
        # replacing NaN age values with the mean age of the data set
        if np.isnan(features[i][0]):
            features[i][0] = mean_age

        # replacing NaN value fares
        if np.isnan(passenger_data['Fare'].values[i]):
            features[i][3] = mean_fare

        # adding binary values of 0 for male and 1 for female passengers
        if sex_array[i] == 'male':
            features[i][1] = 0
        elif sex_array[i] == 'female':
            features[i][1] = 1
        else:
            raise ValueError('correct sex string not found')

    features[:, 0] = features[:, 0] / max_age

    # TODO convert to multi class binary features
    # adding scaled passenger class
    features[:, 2] = passenger_data['Pclass'].values / 3


    # adding scaled number of siblings and spouses
    features[:, 4] = passenger_data['SibSp'].values / np.max(passenger_data['SibSp'].values)

    # adding scaled number of parents and children
    features[:, 5] = passenger_data['Parch'].values / np.max(passenger_data['Parch'].values)

    # adding a binary value for whether each passenger had a cabin or not
    features[:, 6] = pd.isnull(passenger_data['Cabin'].values)

    # building a derived feature of the number of total family members on board
    features[:, 7] = (features[:, 4] + features[:, 5]) / 2

    # normalized ticket number
    ticket_numbers = clean_ticket_numbers(passenger_data['Ticket'].values)
    normalized_ticket_numbers = ticket_numbers / np.nanmax(ticket_numbers)
    features[:, 8] = normalized_ticket_numbers

    # surname is Miss

    # surname is Mr

    # surname is Mrs


    return features


def main():
    all_data = pd.read_csv('./data/train.csv')

    rows = len(all_data.index)
    divider_row = int(rows * 0.99)
    train_data = all_data[:divider_row]
    test_data = all_data[divider_row:rows]

    # print('train_data = \n%s' % str(train_data))
    # print('test_data = \n%s' % str(test_data))

    train_features = construct_features(train_data)
    train_survived_labels = train_data['Survived'].values
    
    test_features = construct_features(test_data)
    test_survived_labels = test_data['Survived'].values

    # print(train_data)
    print(train_features)
    # print(train_survived_labels)

    _, number_of_features = np.shape(train_features)

    # building nn topology
    # TODO experiment with adding regularization
    model = Sequential()
    model.add(Dense(units=20, activation='relu', input_dim=number_of_features))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1, activation='relu'))

    optimizer = optimizers.sgd(lr=0.01)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    epoch = 0

    # hold historical training and test accuracy
    train_accuracy = {}
    test_accuracy = {}

    while epoch < 6000:
        model.fit(train_features, train_survived_labels, epochs=1, batch_size=128)
        test_accuracy[epoch] = model.evaluate(test_features, test_survived_labels, batch_size=128)[1]
        train_accuracy[epoch] = model.evaluate(train_features, train_survived_labels, batch_size=128)[1]

        # TODO add sequential model saving

        print('\nepoch = %i\n' % epoch)

        epoch += 1

    # plotting training and test accuracy histories
    plt.plot(train_accuracy.keys(), train_accuracy.values(), label='train')
    plt.plot(test_accuracy.keys(), test_accuracy.values(), label='test')
    axes = plt.gca()
    axes.set_ylim([0.8, 0.90])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    submission_data = pd.read_csv('./data/test.csv')
    submission_features = construct_features(submission_data)
    print('submission_features = %s' % str(np.shape(submission_features)))
    print('train features = %s' % str(np.shape(train_features)))

    submission_predictions = model.predict(submission_features, batch_size=128)
    # print(submission_predictions)

    comparison_array = (0.5 * np.ones(shape=np.shape(submission_predictions))).astype(dtype=np.float32)
    survived = np.greater(submission_predictions.astype(dtype=np.float32), comparison_array)

    submission_output = pd.DataFrame(index=submission_data.index)
    submission_output['PassengerId'] = submission_data['PassengerId']
    submission_output['Survived'] = survived.astype(dtype=int)
    submission_output.to_csv('./submission.csv', index=False)


if __name__ == '__main__':
    main()
