import pickle
import tensorflow as tf
# import Keras layers you need here
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags

flags.DEFINE_integer('ec', 20, "The number of epochs.")
flags.DEFINE_integer('bs', 64, "The batch size.")
flags.DEFINE_string('ds', '', "Dataset")
flags.DEFINE_string('a', '', "Architecture")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    x_train = train_data['features']
    y_train = train_data['labels']
    x_val = validation_data['features']
    y_val = validation_data['labels']

    return x_train, y_train, x_val, y_val


def main(_):
    # load bottleneck data

    tr_file = 'input/{}_{}_100_bottleneck_features_train.p'.format(FLAGS.a, FLAGS.ds)
    v_file = 'input/{}_{}_bottleneck_features_validation.p'.format(FLAGS.a, FLAGS.ds)


    x_train, y_train, x_val, y_val = load_bottleneck_data(tr_file, v_file)

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)

    nb_classes = len(np.unique(y_train))

    # define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    input_shape = x_train.shape[1:]
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(inp, x)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train model
    model.fit(x_train, y_train,
              nb_epoch=FLAGS.ec,
              batch_size=FLAGS.bs,
              validation_data=(x_val, y_val),
              shuffle=True)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
