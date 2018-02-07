from __future__ import division
import pandas as pd
import numpy as np
from nn import construct_features


def main():
    train_data = pd.read_csv('./data/train.csv')
    submission_data = pd.read_csv('./data/test.csv')

    # train_features = construct_features(train_data)
    # submission_features = construct_features(submission_data)

    # print(train_features)
    # print(submission_features)


    print(train_data)
    #
    # survivors = int(np.sum(data['Survived'].values))
    # passengers = len(data['Age'].values)
    #
    # print(survivors)
    # print(passengers)
    #
    # print('fraction of survivors = %.2f' % (survivors / passengers))


if __name__ == '__main__':
    main()