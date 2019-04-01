from __future__ import print_function
import matplotlib
import os
import pdb
import random
import numpy as np
import tensorflow as tf
from compress import SampleSelection
import transfer_learning
import visualize
matplotlib.use('Agg')

global_seed = 25325
test_idx = 45
epsilon = 0.001

# set random seed
np.random.seed(global_seed)
random.seed(global_seed)
tf.set_random_seed(global_seed)

# NOTE: if you want to add noised testing image to the training set, run "prepare_transfer_values(test_id=test_idx)",
# Otherwise just use "prepare_transfer_values()".
x_train, x_test, y_train, y_test, images_train, cls_train, images_test, cls_test, noise = transfer_learning.prepare_transfer_values(test_id=test_idx)

if not os.path.exists('./embedding.npz'):
    ss = SampleSelection(hidden_dim=32)
    x_train, x_test = ss.fit_embedding(x_train, x_test, cls_train, cls_test)
    np.savez('./embedding.npz',
             x_train=x_train,
             x_test=x_test)
else:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    f = np.load(str(dir_path) + '/embedding.npz')
    x_train = f['x_train']
    x_test = f['x_test']

s_x = tf.constant(np.array([x_test[test_idx]]), dtype=tf.float32)
s_y = tf.constant(np.array([y_test[test_idx]]), dtype=tf.float32)


def linear_model(x_ph, reuse=False):
    with tf.variable_scope('model') as vs:
        if reuse:
            vs.reuse_variables()

        return tf.layers.dense(x_ph, 2, activation=None)


def mlp_model(x_ph, reuse=False):
    with tf.variable_scope("model") as vs:
        if reuse:
            vs.reuse_variables()
        h1 = tf.layers.dense(x_ph, 8, activation=tf.nn.tanh)
        h2 = h1
        logit = tf.layers.dense(h2, 2, activation=None)

    return logit


model = linear_model

maxiter = 30
n_class = 2
X = tf.placeholder(tf.float32, [None, x_train.shape[1]], name="X")
Y = tf.placeholder(tf.float32, [None, 2], name="Y")

logit = model(X, reuse=False)
logit_ent = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y)
loss = tf.reduce_mean(logit_ent)

# First lets try trainditional optimizer
optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='SLSQP', options={'maxiter': maxiter})
print("maxiter=%d" % (maxiter))

train_variables = tf.trainable_variables()
flat_vars = tf.concat([tf.reshape(var, [-1]) for var in train_variables], axis=0)

feed_dict = {X: x_train, Y: y_train}
test_fd = {X: x_test, Y: y_test}
correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
category_idx, probability = [], []


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    optimizer.minimize(sess, feed_dict=feed_dict)
    best_loss = sess.run(loss, feed_dict)
    b_lent = sess.run(logit_ent, feed_dict=feed_dict)

    print("Training accuracy: %f" % (accuracy.eval({X: x_train, Y: y_train})))
    print("loss = %f" % (loss.eval(feed_dict)))
    s_logit = model(s_x, reuse=True)
    test_point_softmax = tf.nn.softmax(s_logit)
    softmax_result = test_point_softmax.eval(session=sess)[0]

    if softmax_result[0] > softmax_result[1]:
        category_idx.append(0)
    else:
        category_idx.append(1)

    print("Test acc %f, loss %f" % (accuracy.eval(test_fd), loss.eval(test_fd)))

    if np.argmax(y_test[test_idx]) == 0:
        inequalities = [- ((s_logit[0][0] - s_logit[0][1]) + epsilon)]
        index = 0
    else:
        inequalities = [- ((s_logit[0][1] - s_logit[0][0]) + epsilon)]
        index = 1

    probability.append("%.2f" % round(softmax_result[index], 2))

    sess.run(tf.global_variables_initializer())
    feed_dict = {X: x_train, Y: y_train}

    new_ops = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                     inequalities=inequalities,
                                                     method='SLSQP',
                                                     options={'maxiter': maxiter})

    new_ops.minimize(sess, feed_dict=feed_dict)
    cur_loss = sess.run(loss, feed_dict)
    print('New loss %f' % (cur_loss))
    print("Training accuracy: %f" % (accuracy.eval(feed_dict)))
    print(tf.nn.softmax(s_logit).eval(), sess.run(inequalities))

    hat_lent = b_lent
    rdata_x = []
    rdata_y = []
    noise_mask = []

    # starting optimization process
    for tt in range(5):
        print("Start iter=%d------------------------------------" % (tt))

        # constraint one
        cur_ent = logit_ent.eval(feed_dict)
        idx = np.argmax((cur_ent - hat_lent), axis=0)

        rdata_x.append(images_train[idx])
        rdata_y.append(cls_train[idx])
        noise_mask.append(noise[idx])
        print("selecting data x=%s, y_train=%s, diff =%s" % (x_train[idx], y_train[idx], cur_ent[idx] - hat_lent[idx]))

        # delete data points from dataset
        x_train = np.delete(x_train, idx, axis=0)
        y_train = np.delete(y_train, idx, axis=0)
        images_train = np.delete(images_train, idx, axis=0)
        cls_train = np.delete(cls_train, idx, axis=0)
        noise = np.delete(noise, idx, axis=0)
        feed_dict = {X: x_train, Y: y_train}

        sess.run(tf.global_variables_initializer())
        optimizer.minimize(sess, feed_dict=feed_dict)
        best_loss = sess.run(loss, feed_dict)
        hat_lent = sess.run(logit_ent, feed_dict=feed_dict)
        print("Test acc %f, loss %f" % (accuracy.eval(test_fd), loss.eval(test_fd)))
        print("Unconstrained iter %d loss: %f, acc: %f" % (
        tt, loss.eval(feed_dict), accuracy.eval({X: x_train, Y: y_train})))
        s_logit = model(s_x, reuse=True)
        test_point_softmax = tf.nn.softmax(s_logit)
        softmax_result = test_point_softmax.eval(session=sess)[0]

        if softmax_result[0] > softmax_result[1]:
            category_idx.append(0)
        else:
            category_idx.append(1)
        probability.append("%.2f" % round(softmax_result[index], 2))

        sess.run(tf.global_variables_initializer())
        new_ops.minimize(sess, feed_dict=feed_dict)
        cur_loss = sess.run(loss, feed_dict)
        print("Constrained Iter %d loss %f accuracy: %f" % (tt, cur_loss, accuracy.eval(feed_dict)))
        print("Test acc %f, loss %f" % (accuracy.eval(test_fd), loss.eval(test_fd)))
        print(tf.nn.softmax(s_logit).eval(), sess.run(inequalities))
        print('best_loss: ', best_loss)
        print('cur_loss: ', cur_loss)
        if best_loss >= cur_loss:
            break

# TODO: Change the directory.
visualize.plot_images(path='/Users/hanxing/Desktop/figure', images=rdata_x, cls_true=rdata_y,
                      test_image=images_test[test_idx], test_label=cls_test[test_idx], category_idx=category_idx,
                      probability=probability, test_idx=test_idx, noise=noise_mask)
