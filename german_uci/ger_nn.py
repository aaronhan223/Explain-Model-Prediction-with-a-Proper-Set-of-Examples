from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os 
import pdb
import copy
import random
import itertools
import numpy as np 
import tensorflow as tf 

from tqdm import tqdm
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
global_seed = 25325



test_idx = 208
epsilon = 0.001

# set random seed
np.random.seed(global_seed)
random.seed(global_seed)
tf.set_random_seed(global_seed)
hd_str = 'Creditability,Account Balance,Duration of Credit \
        (month),Payment Status of Previous Credit,Purpose,Credit Amount,Value Savings/Stocks,Length of current employment,Instalment per cent,Sex & Marital Status,Guarantors,Duration in Current address,Most valuable available asset,Age (years),Concurrent Credits,Type of apartment,No of Credits at this Bank,Occupation,No of dependents,Telephone,Foreign Worker'

# TODO transfer y to hot vector
# construct models
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# load dataset
dataset = np.loadtxt('german_credit.csv', delimiter=',', skiprows=1)
data_X = dataset[:, 1:].astype(np.float)
mean_X = np.mean(data_X, axis=0)
std_X = np.std(data_X, axis=0)
data_X = (data_X - mean_X) / std_X
data_y = dataset[:, 0]
data_y = np.eye(2)[data_y.astype(np.int)]
x_train, x_test, y_train, y_test = train_test_split(data_X, data_y)
x_train , y_train = shuffle(x_train, y_train)

s_x = tf.constant(np.array([x_test[test_idx]]), dtype=tf.float32)
s_y = tf.constant(np.array([y_test[test_idx]]), dtype=tf.float32)



def plot_data(name, x_train, y_train,  nn_x = None, rdata=None):
    # plot visualization figure
    plt.figure(figsize=[4, 4])
    # plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    yy = np.argmax(y_train, axis=1)
    pos = np.where(yy == 1)
    neg = np.where(yy == 0)

    plt.scatter(x_train[pos, 0], x_train[pos, 1], marker='o', c='b',s=5**2)
    plt.scatter(x_train[neg, 0], x_train[neg, 1], marker='x', c='r',s=5**2)
    
    if nn_x is not None:
         plt.scatter(nn_x[:, 0], nn_x[:, 1], marker='.', c='k', s=5**2)

    if rdata is not None and len(rdata) > 0:
        rdata = np.array(rdata)
        plt.scatter(data_X[test_idx, 0], data_X[test_idx, 1], marker='o', c='c',s=10**2)
        plt.scatter(rdata[:, 0], rdata[:, 1], marker='>', c='m', s=10**2)
    
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    plt.savefig(name, bbox_inches='tight')
    plt.clf()

def linear_model(x_ph, reuse=False):
    with tf.variable_scope('model') as vs:
        if reuse:
            vs.reuse_variables()
        
        return tf.layers.dense(x_ph, 2, activation=None)

def mlp_model(x_ph, reuse=False):
    with tf.variable_scope("model") as vs:
        if reuse:
            vs.reuse_variables()
        h1 = tf.layers.dense(x_ph, 4, activation=tf.nn.tanh)
        h2 = tf.layers.dense(h1, 4, activation=tf.nn.tanh)
        # h2 = h1
        logit = tf.layers.dense(h2, 2, activation=None)
    
    return logit

model = mlp_model

x1 = np.arange(-0.1, 1.1, 0.001)
x2 = np.arange(-0.1, 1.1, 0.001)
margin_X = np.array(list(itertools.product(x1, x2)))


maxiter = 30
# linear
# maxiter = 50
n_class = 2
X = tf.placeholder(tf.float32, [None, x_train.shape[1]], name="X")
Y = tf.placeholder(tf.float32, [None, 2], name="Y")


 
logit = model(X, reuse=False)
logit_ent = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y)
loss = tf.reduce_mean(logit_ent)

# First lets try trainditional optimizer
optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, 
            method='SLSQP',
            options={'maxiter': maxiter})
print("maxiter=%d"%(maxiter))

train_variables = tf.trainable_variables()
flat_vars = tf.concat([tf.reshape(var, [-1]) for var in train_variables], axis=0)


feed_dict = {X:x_train, Y:y_train}
test_fd = {X:x_test, Y:y_test}
correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.to_float(correct_prediction))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    optimizer.minimize(sess, feed_dict=feed_dict)
    best_loss = sess.run(loss, feed_dict)
    b_lent = sess.run(logit_ent, feed_dict=feed_dict)

    print("Training accuracy: %f"%(accuracy.eval({X:x_train, Y:y_train})))
    print("loss = %f"%(loss.eval(feed_dict)))
    s_logit = model(s_x, reuse=True)

    print("Test acc %f, loss %f" %(accuracy.eval(test_fd), loss.eval(test_fd)))


    # plot_data("hpnn/opt.pdf", x_train, y_train, nn_x=nn_x)
    # tf.nn.softmax_cross_entropy_with_logits(logits=model(s_x, reuse=True), labels=s_y)
    # TODO check label for s_y
    # lets do the ugly part
    # inequalities
    if np.argmax(y_test[test_idx]) == 0:
        inequalities = [- ((s_logit[0][0] - s_logit[0][1]) + epsilon)]
    else:
        inequalities = [- ((s_logit[0][1] - s_logit[0][0]) + epsilon)]
    pdb.set_trace()
    # Here is the new dataset
    
    sess.run(tf.global_variables_initializer())
    # x_train = np.delete(x_train, test_idx, axis=0)
    # y_train = np.delete(y_train, test_idx, axis=0)
    feed_dict = {X:x_train, Y:y_train}

    new_ops = tf.contrib.opt.ScipyOptimizerInterface(loss, 
            inequalities=inequalities, 
            method='SLSQP',
            options={'maxiter': maxiter })
    
    new_ops.minimize(sess, feed_dict=feed_dict)
    cur_loss = sess.run(loss, feed_dict)
    print('New loss %f'%(cur_loss))
    #TODO after accuracy
    print("Training accuracy: %f"%(accuracy.eval(feed_dict)))
    print(tf.nn.softmax(s_logit).eval(), sess.run(inequalities))
    # nn_x = fine_marginline(logit)
    # plot_data("hpnn/opt_1.pdf", x_train, y_train, nn_x=nn_x)
    hat_lent = b_lent # np.delete(b_lent, test_idx, axis=0)
    rdata_x = []
    rdata_y = []
    pdb.set_trace()
    # starting optimization process
    for tt in range(50):
        print("Start iter=%d------------------------------------"%(tt))
        
        # constraint one
        cur_ent = logit_ent.eval(feed_dict)
        idx = np.argmax((cur_ent - hat_lent), axis=0)

        rdata_x.append(x_train[idx])
        rdata_y.append(y_train[idx])
        print("selecting data x=%s, y_train=%s, diff =%s"%(x_train[idx], y_train[idx], cur_ent[idx] - hat_lent[idx]))

        # delete data points from dataset
        x_train = np.delete(x_train, idx, axis=0)
        y_train = np.delete(y_train, idx, axis=0)
        # hat_lent = np.delete(hat_lent, idx, axis=0)
        feed_dict={X:x_train, Y:y_train}

        sess.run(tf.global_variables_initializer())
        optimizer.minimize(sess, feed_dict=feed_dict)
        best_loss = sess.run(loss, feed_dict)
        hat_lent = sess.run(logit_ent, feed_dict=feed_dict)
        print("Test acc %f, loss %f" %(accuracy.eval(test_fd), loss.eval(test_fd)))
        print("Unconstrain iter %d loss: %f, acc: %f"%(tt,loss.eval(feed_dict),  accuracy.eval({X:x_train, Y:y_train})))
        s_logit = model(s_x, reuse=True)

        # first one
       #  nn_x = fine_marginline(logit)
        # plot_data("hpnn/demo_%d_0.pdf"%(tt), x_train, y_train, nn_x=nn_x)

        # second one
        # sess.run(tf.global_variables_initializer())
        new_ops.minimize(sess, feed_dict=feed_dict)
        cur_loss = sess.run(loss, feed_dict)
        print("Constrained Iter %d loss %f accuracy: %f"%(tt, cur_loss, accuracy.eval(feed_dict)))
        print("Test acc %f, loss %f" %(accuracy.eval(test_fd), loss.eval(test_fd)))
        # nn_x = fine_marginline(logit)
        # plot_data("hpnn/demo_%d_1.pdf"%(tt), x_train, y_train, nn_x=nn_x, rdata=rdata_x)
        print(tf.nn.softmax(s_logit).eval(), sess.run(inequalities))
        
        if  best_loss >= cur_loss :
            # save to file
            rdata_x.append(x_test[test_idx])
            rdata_y.append(y_test[test_idx])

            rdata_x = np.array(rdata_x, dtype=np.float)
            rdata_x = rdata_x * std_X + mean_X
            rdata_x = np.array(rdata_x, dtype=np.int)
            rdata_y = np.array(rdata_y, dtype=np.int)
            rdata_y = np.expand_dims(np.argmax(rdata_y, axis=1), axis=1)
        
            data = np.concatenate([rdata_y, rdata_x], axis=1)

            np.savetxt('nn_select.csv', data, fmt='%d', delimiter=',', header=hd_str)
            break

