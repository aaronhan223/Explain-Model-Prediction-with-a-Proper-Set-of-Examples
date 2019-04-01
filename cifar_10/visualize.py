from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pdb
import cifar10

h = .02  # step size in the mesh


def viz(score, Z, iter, X_train, y_train, X_test, y_test, xx, yy, input_x, input_y, sample=None, label=None):
    '''
    Visualization for half moon data set.
    '''

    names = ["Selection # {}, test point red, selected sample white".format(iter)]

    figure = plt.figure(figsize=(9, 3))
    i = 1

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FFF000', '#000FFF'])
    ax = plt.subplot(1, 2, i)
    ax.set_title("Input data")
    # Plot the training points
    ax.scatter(input_x[:, 0], input_x[:, 1], c=input_y, cmap=cm_bright, edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.5, edgecolors='r')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers (multiple)
    for name in names:

        ax = plt.subplot(1, 2, i)
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='r', alpha=0.5)

        if sample is not None:
            ax.scatter(sample[:, 0], sample[:, 1], c=label, cmap=cm_bright, edgecolors='w')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
        i += 1

    plt.tight_layout()
    plt.savefig('/Users/hanxing/Desktop/figure/half_moon_idx_new_iter_{}.png'.format(iter))


def plot_images(path, images, cls_true, test_image, test_label, category_idx, probability, test_idx, noise):
    '''
    Cifar 10 data set visualization: constrained optimization
    '''
    assert len(images) == len(cls_true)
    class_names = cifar10.load_class_names()
    fig, axes = plt.subplots(1, len(noise) + 1, figsize=(3 * (len(noise) + 1), len(noise) + 1))
    plt.subplots_adjust(wspace=0.1)
    fontsize = 14

    for i, ax in enumerate(axes.flat):

        if i == 0:
            ax.imshow(test_image)
            ax.set_xlabel('Test image \n {}'.format(class_names[test_label]), fontsize=fontsize)
        else:
            ax.imshow(images[i-1])
            if noise[i-1] is not None:
                ax.set_xlabel('Train image \n {} \n {}'.format(class_names[cls_true[i-1]], noise[i-1]),
                              fontsize=fontsize)
            else:
                ax.set_xlabel('Train image \n {}'.format(class_names[cls_true[i - 1]]), fontsize=fontsize)
        ax.set_xticks([])
        ax.set_yticks([])

    # fig.suptitle("Constrained Optimization", fontsize=20)
    plt.savefig(path + '/con_{}.png'.format(test_idx))


def plot_influence_image(path, x_train, x_test, images_train, y_train, images_test, y_test, noise):
    '''
    Cifar 10 data set visualization: influence function
    '''

    sns.set(color_codes=True)
    class_names = cifar10.load_class_names()
    f = np.load('output/influence_results.npz')
    test_idx = int(f['test_idx'])
    distances = f['distances']
    # flipped_idx = f['flipped_idx']
    inception_Y_pred_correct = f['inception_Y_pred_correct']
    inception_predicted_loss_diffs = f['inception_predicted_loss_diffs']
    sns.set_style('white')
    num_data = 5
    fontsize = 14

    print('Test image:')
    print(y_test[test_idx], inception_Y_pred_correct[test_idx])
    fig, axes = plt.subplots(1, num_data + 1, figsize=(3 * (num_data + 1), num_data + 1))
    plt.subplots_adjust(wspace=0.1)
    axes[0].imshow(images_test[test_idx])
    axes[0].set_xlabel('Test image \n {}'.format(class_names[y_test[test_idx]]), fontsize=fontsize)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    print('Top from Inception:')
    for counter, train_idx in enumerate(np.argsort(inception_predicted_loss_diffs)[-5:]):
        print(train_idx, y_train[train_idx], distances[train_idx], inception_predicted_loss_diffs[train_idx])
        x_train = images_train[train_idx, :]
        axes[counter+1].imshow(x_train)
        if noise[train_idx] is not None:
            axes[counter+1].set_xlabel('Train image \n {} \n {}'.format(class_names[y_train[train_idx]],
                                                                        noise[train_idx]), fontsize=fontsize)
        else:
            axes[counter + 1].set_xlabel('Train image \n {}'.format(class_names[y_train[train_idx]]), fontsize=fontsize)
        axes[counter+1].set_xticks([])
        axes[counter+1].set_yticks([])

    # fig.suptitle("Influence Function", fontsize=20)
    # plt.tight_layout()
    plt.savefig(path + '/inf_{}.png'.format(test_idx))
