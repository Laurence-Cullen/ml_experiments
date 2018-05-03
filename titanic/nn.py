from __future__ import division
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense
from keras import regularizers
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def clean_ticket_numbers(ticket_strings):
    """
    Strips out any non numeric prefixes to ticket number strings.
    """
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
    features = pd.DataFrame()

    features['age'] = passenger_data['Age'] / passenger_data['Age'].max()

    # fill in nan values with mean of normalized age
    features['age'] = features['age'].fillna(features['age'].mean())

    def sex_to_binary(sex):
        if sex == 'male':
            return 1
        elif sex == 'female':
            return 0
        elif math.isnan(sex):
            return 0.5
        else:
            raise ValueError('error mapping sex binary values')

    features['is_male'] = passenger_data['Sex'].map(sex_to_binary)

    # adding scaled passenger fare
    features['fare'] = passenger_data['Fare'] / passenger_data['Fare'].max()

    # adding scaled passenger class
    features['p_class'] = passenger_data['Pclass'] / 3

    # adding scaled number of siblings and spouses
    features['sib_spouse'] = passenger_data['SibSp'].values / passenger_data['SibSp'].max()

    # adding scaled number of parents and children
    features['par_child'] = passenger_data['Parch'] / passenger_data['Parch'].max()

    # adding a binary value for whether each passenger had a cabin or not
    features['has_cabin'] = pd.notnull(passenger_data['Cabin']).values.astype(float)

    # building a derived feature of the number of total family members on board
    features['kin_on_board'] = (features['sib_spouse'] + features['par_child']) / 2

    # normalized ticket number
    ticket_numbers = clean_ticket_numbers(passenger_data['Ticket'].values)
    normalized_ticket_numbers = ticket_numbers / np.nanmax(ticket_numbers)
    features['ticket_num'] = normalized_ticket_numbers

    # title is Miss
    features['miss'] = passenger_data['Name'].str.contains('Miss').values.astype(float)

    # title is Mr
    features['mr'] = passenger_data['Name'].str.contains('Mr').values.astype(float)

    # title is Mrs
    features['mrs'] = passenger_data['Name'].str.contains('Mrs').values.astype(float)

    # title is Master
    features['master'] = passenger_data['Name'].str.contains('Master').values.astype(float)

    # # embarked at C
    # features['emb_c'] = passenger_data['Embarked'].str.contains('C').values.astype(float)
    #
    # # embarked at S
    # features['emb_s'] = passenger_data['Embarked'].str.contains('S').values.astype(float)
    #
    # # embarked at Q
    # features['emb_q'] = passenger_data['Embarked'].str.contains('Q').values.astype(float)

    def cabin_to_float(cabin):
        cabin = str(cabin)
        if 'nan' in cabin:
            return 0
        elif 'A' in cabin:
            return 1
        elif 'B' in cabin:
            return 5 / 6
        elif 'C' in cabin:
            return 4 / 6
        elif 'D' in cabin:
            return 3 / 6
        elif 'E' in cabin:
            return 2 / 6
        elif 'F' in cabin:
            return 1 / 6
        elif 'G' in cabin:
            return 0
        else:
            return 0

    features['level'] = passenger_data['Cabin'].map(cabin_to_float)

    features.astype(float)
    return features


def main():
    all_data = pd.read_csv('./data/train.csv')

    rows = len(all_data.index)
    divider_row = int(rows * 0.80)
    train_data = all_data[:divider_row]
    test_data = all_data[divider_row:rows]

    train_features = construct_features(train_data)
    train_features.to_csv(path_or_buf='full_features.csv')

    print(train_features.values)

    train_survived_labels = train_data['Survived'].values
    
    test_features = construct_features(test_data)
    test_survived_labels = test_data['Survived'].values

    _, number_of_features = np.shape(train_features)

    reg_value = 0.000

    # building nn topology
    model = Sequential()
    model.add(Dense(units=15,
                    activation='relu',
                    input_dim=number_of_features,
                    kernel_regularizer=regularizers.l2(reg_value)))

    model.add(Dense(units=10, activation='relu', kernel_regularizer=regularizers.l2(reg_value)))
    model.add(Dense(units=5, activation='relu', kernel_regularizer=regularizers.l2(reg_value)))
    # model.add(Dense(units=5, activation='relu', kernel_regularizer=regularizers.l2(reg_value)))
    # model.add(Dense(units=5, activation='relu', kernel_regularizer=regularizers.l2(reg_value)))
    model.add(Dense(units=1, activation='sigmoid'))

    optimizer = optimizers.sgd(lr=0.001)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    epochs = 20000

    history = model.fit(x=train_features.values,
                        y=train_survived_labels,
                        validation_data=(test_features, test_survived_labels),
                        epochs=epochs,
                        batch_size=64)

    plt.plot(history.history['acc'], label='train accuracy')
    plt.plot(history.history['val_acc'], label='test accuracy')

    axes = plt.gca()
    axes.set_ylim([0.75, 0.90])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    submission_data = pd.read_csv('./data/test.csv')
    submission_features = construct_features(submission_data)
    print('submission_features = %s' % str(np.shape(submission_features)))
    print('train features = %s' % str(np.shape(train_features)))

    submission_predictions = model.predict(submission_features, batch_size=128)

    comparison_array = (0.5 * np.ones(shape=np.shape(submission_predictions))).astype(dtype=np.float32)
    survived = np.greater(submission_predictions.astype(dtype=np.float32), comparison_array)

    submission_output = pd.DataFrame(index=submission_data.index)
    submission_output['PassengerId'] = submission_data['PassengerId']
    submission_output['Survived'] = survived.astype(dtype=int)
    submission_output.to_csv('./submission.csv', index=False)

    print('submissions saved to ./submissions.csv')


if __name__ == '__main__':
    main()
