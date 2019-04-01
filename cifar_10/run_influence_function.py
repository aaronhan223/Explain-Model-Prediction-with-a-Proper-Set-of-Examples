from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
import influence.dataset as dataset
from influence.dataset import DataSet
import random
import transfer_learning
from compress import SampleSelection
import os
import visualize
import pdb


def get_Y_pred_correct_inception(model):
    Y_test = model.data_sets.test.labels
    if np.min(Y_test) < -0.5:
        Y_test = (np.copy(Y_test) + 1) / 2
    Y_pred = model.sess.run(model.preds, feed_dict=model.all_test_feed_dict)
    Y_pred_correct = np.zeros([len(Y_test)])
    for idx, label in enumerate(Y_test):
        Y_pred_correct[idx] = Y_pred[idx, int(label)]
    return Y_pred_correct


global_seed = 25325
# 35
# 45 25 for add noise
test_idx = 45

np.random.seed(global_seed)
random.seed(global_seed)
tf.set_random_seed(global_seed)

num_classes = 2

# load dataset
x_train, x_test, _, _, images_train, y_train, images_test, y_test, noise = transfer_learning.prepare_transfer_values(test_id=test_idx)

if not os.path.exists('./embedding.npz'):
    ss = SampleSelection(hidden_dim=32)
    x_train, x_test = ss.fit_embedding(x_train, x_test, y_train, y_test)
    np.savez('./embedding.npz',
             x_train=x_train,
             x_test=x_test)
else:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    f = np.load(str(dir_path) + '/embedding.npz')
    x_train = f['x_train']
    x_test = f['x_test']

train = DataSet(x_train, y_train)
test = DataSet(x_test, y_test)
validation = None

data_sets = base.Datasets(train=train, validation=validation, test=test)

input_dim = 32
weight_decay = 0.001
batch_size = x_train.shape[0]
initial_learning_rate = 0.001
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000
num_classes = 2

tf.reset_default_graph()

inception_model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='influence_function')

inception_model.train()

inception_predicted_loss_diffs = inception_model.get_influence_on_test_loss(
    [test_idx],
    np.arange(len(inception_model.data_sets.train.labels)),
    force_refresh=True)

X_test = x_test[test_idx, :]
Y_test = y_test[test_idx]

distances = dataset.find_distances(X_test, x_train)
flipped_idx = y_train != Y_test
inception_Y_pred_correct = get_Y_pred_correct_inception(inception_model)

np.savez(
    'output/influence_results.npz',
    test_idx=test_idx,
    distances=distances,
    flipped_idx=flipped_idx,
    inception_Y_pred_correct=inception_Y_pred_correct,
    inception_predicted_loss_diffs=inception_predicted_loss_diffs)

# TODO: Change the directory.
visualize.plot_influence_image(path='/Users/hanxing/Desktop/figure', x_train=x_train, x_test=x_test, y_train=y_train,
                               y_test=y_test, images_train=images_train, images_test=images_test, noise=noise)
