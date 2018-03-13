import urllib.request as urllib2
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
print('in directory %s' % dir_path)
working_directory = os.path.dirname(__file__)

train_resource = urllib2.urlopen('https://pjreddie.com/media/files/mnist_train.csv')
print('train resource downloaded')
train_csv_string = str(train_resource.read())
train_save_path = './data/train.csv'
train_file = open(os.path.join(working_directory, train_save_path), mode='w')
train_file.write(train_csv_string)
print('training data file written to %s' % train_save_path)

test_resource = urllib2.urlopen('https://pjreddie.com/media/files/mnist_test.csv')
print('test resource downloaded')
test_csv_string = str(test_resource.read())
test_save_path = './data/test.csv'
test_file = open(os.path.join(working_directory, test_save_path), mode='w')
test_file.write(test_csv_string)
print('testing data file written to %s' % test_save_path)
