import os
import inception
import cifar10
from inception import transfer_values_cache
import numpy as np
import tensorflow as tf
import random
import pdb


global_seed = 25325
np.random.seed(global_seed)
random.seed(global_seed)

num_of_training = 1000
num_of_testing = 200
noise_list = random.sample(range(1000), 20)
noise_var = 1e-8


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def prepare_transfer_values(test_id=None):

    # Download the CIFAR 10 data
    cifar10.maybe_download_and_extract()
    class_names = cifar10.load_class_names()
    print("Class names for CIFAR10 is: ", class_names)

    images_train, cls_train, labels_train = cifar10.load_training_data()
    images_test, cls_test, labels_test = cifar10.load_test_data()

    if not os.path.exists('./index_memory.npz'):
        print('Generate random index ...')
        cls_0_idx_train = np.squeeze(np.argwhere(cls_train == 0))[:int(num_of_training / 2)]
        cls_1_idx_train = np.squeeze(np.argwhere(cls_train == 1))[:int(num_of_training / 2)]
        cls_0_idx_test = np.squeeze(np.argwhere(cls_test == 0))[:int(num_of_testing / 2)]
        cls_1_idx_test = np.squeeze(np.argwhere(cls_test == 1))[:int(num_of_testing / 2)]

        cls_idx_train = np.concatenate((cls_0_idx_train, cls_1_idx_train))
        cls_idx_test = np.concatenate((cls_0_idx_test, cls_1_idx_test))

        np.random.shuffle(cls_idx_train)
        np.random.shuffle(cls_idx_test)

        np.savez('./index_memory.npz',
                 cls_idx_train=cls_idx_train,
                 cls_idx_test=cls_idx_test)

    else:
        print('Index already generated.')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        f = np.load(str(dir_path) + '/index_memory.npz')
        cls_idx_train = f['cls_idx_train']
        cls_idx_test = f['cls_idx_test']

    images_train, cls_train, labels_train = images_train[cls_idx_train], cls_train[cls_idx_train], labels_train[cls_idx_train]
    images_test, cls_test, labels_test = images_test[cls_idx_test], cls_test[cls_idx_test], labels_test[cls_idx_test]
    labels_train, labels_test = labels_train[:, :2], labels_test[:, :2]
    noise_marker = [None] * num_of_training

    print("Size of:")
    print("- Training-set:\t\t{}".format(len(images_train)))
    print("- Testing-set:\t\t{}".format(len(images_test)))

    # Download the inception model
    inception.maybe_download()
    model = inception.Inception()

    file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
    file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')

    print("Processing Inception transfer-values for training-images ...")

    # Scale images because Inception needs pixels to be between 0 and 255,
    # while the CIFAR-10 functions return pixels between 0.0 and 1.0
    images_scaled = images_train * 255.0

    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train, images=images_scaled, model=model)

    print("Processing Inception transfer-values for test-images ...")
    images_scaled = images_test * 255.0
    transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test, images=images_scaled, model=model)

    if test_id is not None:
        all_noise_test = np.zeros((len(noise_list), 32, 32, 3))
        for idx, noise in enumerate(noise_list):
            tf.set_random_seed(noise)
            all_noise_test[idx] = gaussian_noise_layer(images_test[test_id], std=noise_var).eval(session=tf.Session())
            noise_marker.append('Original image with noise')

        images_train = np.vstack((images_train, all_noise_test))
        cls_train = np.append(cls_train, np.repeat(cls_test[test_id], len(noise_list)))
        labels_train = np.vstack(
            (labels_train, np.repeat(np.expand_dims(labels_test[test_id], axis=0), len(noise_list), axis=0)))
        file_path_cache_noise = os.path.join(cifar10.data_path, 'inception_cifar10_noise.pkl')

        print("Processing Inception transfer-values for noised training-images ...")
        noise_scaled = all_noise_test * 255.0
        transfer_values_noise = transfer_values_cache(cache_path=file_path_cache_noise, images=noise_scaled, model=model)
        transfer_values_train = np.vstack((transfer_values_train, transfer_values_noise))

    return transfer_values_train, transfer_values_test, labels_train, labels_test, images_train, cls_train, images_test, cls_test, noise_marker
