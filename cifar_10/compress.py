from __future__ import print_function
import torch
import random, pdb
import numpy as np
import torch.nn as nn
import torch.optim as optim
import InputDataset
from torch.utils.data import DataLoader
from scipy.optimize import fmin_slsqp, fmin_cobyla
from training_function import train, test, test_con, computer_optim_loss
from training_function import minimize as minimize_
from sklearn.metrics import mean_squared_error
from visualize import viz
from scipy.optimize import minimize

global_seed = 2532

# test_idx = 3

np.random.seed(global_seed)
random.seed(global_seed)
torch.manual_seed(global_seed)


class FeedForwardNeuralNet(nn.Module):
    '''
    Feedforward network on top of Inception v3: 2048 to 64 to 2.
    '''
    def __init__(self, input_size, hidden_dim, output_dim=2):
        super(FeedForwardNeuralNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.non_linear = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        out = self.fc1(x)
        out = self.non_linear(out)
        pred = self.fc2(out)

        return pred, out


class LinearModel(nn.Module):
    '''
    linear model on top of Inception v3: 64 to 2.
    '''
    def __init__(self, input_size, output_dim=2):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_size, output_dim, bias=True)

    def forward(self, x):
        pred = self.fc(x)
        return pred


class ConditionalOptimizer:

    def __init__(self, loss, x0, cons):
        self.loss = loss
        self.x0 = x0
        self.cons = cons

    def solve(self):

        fmin_results = fmin_slsqp(
            func=self.loss,
            x0=self.x0,
            f_ieqcons=self.cons,
            acc=1e-4)

        return fmin_results


class SampleSelection:

    def __init__(self, max_data_point=6, hidden_dim=32, num_epochs=300, n_class=2, maxiter=50, epsilon=1e-8):

        self.max_data_point = max_data_point
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.n_class = n_class
        self.maxiter = maxiter
        self.epsilon = epsilon

    def fit_embedding(self, x_train, x_test, cls_train, cls_test):

        model = FeedForwardNeuralNet(input_size=x_train.shape[1], hidden_dim=self.hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_dataset = InputDataset.InputDataset(inputs=x_train, results=cls_train, input_dim=len(x_train), label=np.zeros(len(x_train)))
        train_loader = DataLoader(train_dataset, batch_size=len(x_train), num_workers=4, shuffle=False)
        test_dataset = InputDataset.InputDataset(inputs=x_test, results=cls_test, input_dim=len(x_test), label=np.zeros(len(x_train)))
        test_loader = DataLoader(test_dataset, batch_size=len(x_test), num_workers=4, shuffle=False)
        train_embed, test_embed = np.zeros((x_train.shape[0], self.hidden_dim)), np.zeros((x_test.shape[0], self.hidden_dim))

        print("Learn embedding from 2048 to {} ...".format(self.hidden_dim))
        for epoch_idx in range(1, self.num_epochs + 1):
            total_loss_train, train_embed = train(train_loader, model, optimizer, epoch_idx, train_embed)
            if epoch_idx % 50 == 0:
                print('Epoch [%d]: Train loss: %.3f' % (epoch_idx, total_loss_train))

        for epoch_idx in range(1, self.num_epochs + 1):
            total_loss_test, test_embed = train(test_loader, model, optimizer, epoch_idx, test_embed)
            if epoch_idx % 50 == 0:
                print('Epoch [%d]: Test loss: %.3f' % (epoch_idx, total_loss_test))

        return train_embed, test_embed

    def select_sample(self, X_train, X_test, y_train, y_test, cls_train, cls_test, test_idx, plot, **kwargs):

        if 'images_train' in kwargs: images_train = kwargs.pop('images_train')
        if 'images_test' in kwargs: images_test = kwargs.pop('images_test')
        if 'X' in kwargs:
            X = kwargs.pop('X')
            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

        print("Constrained and unconstrained optimization on embeddings ...")
        test_point_x = torch.FloatTensor(np.array([X_test[test_idx]]))
        test_point_y = torch.FloatTensor(np.array([y_test[test_idx]]))
        idx = np.argwhere(test_point_y[0] == 1)[0][0]
        idx_f = -1 * idx + 1

        # Then do the normal process for 64 dimension data.
        model = LinearModel(input_size=self.hidden_dim, output_dim=self.n_class)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_dataset = InputDataset.InputDataset(inputs=X_train, results=cls_train, label=y_train, input_dim=len(X_train))
        train_loader = DataLoader(train_dataset, batch_size=len(X_train), num_workers=4, shuffle=False)
        test_dataset = InputDataset.InputDataset(inputs=X_test, results=cls_test, label=y_test, input_dim=len(X_test))
        test_loader = DataLoader(test_dataset, batch_size=len(X_test), num_workers=4, shuffle=False)

        removed_data_x, removed_data_y, category_idx, probability, correct_prediction_test = [], [], [], [], np.zeros(X_test.shape[0])
        loss_per_data_point_uncon, correct_prediction, loss_per_data_point_con = np.zeros(X_train.shape[0]), np.zeros(X_train.shape[0]), np.zeros(X_train.shape[0])
        accuracy, train_loss = 0, 0

        for epoch_idx in range(1, self.num_epochs + 1):
            train_loss, loss_per_data_point_uncon, accuracy = minimize_(train_loader, model, optimizer, epoch_idx,
                                                                        loss_per_data_point_uncon, correct_prediction)

        train_loss, loss_per_data_point_uncon, accuracy = test_con(train_loader, model, correct_prediction,
                                                                   loss_per_data_point_uncon)
        print("Unconstrained training acc {}, loss {} ".format(accuracy, train_loss))

        test_loss, test_accuracy = test(test_loader, model, correct_prediction_test)
        print("Unconstrained testing acc {}, loss {} ".format(test_accuracy, test_loss))

        test_point_logit = model(test_point_x)
        fmin_loss_fn = self.get_train_fmin_loss_fn(model, train_loader)
        inequality_fn = self.get_inequality_fn(test_point_logit, idx, idx_f, self.epsilon, model)
        x0 = np.random.uniform(-0.5, 0.5, 2 * (self.hidden_dim + 1))

        new_ops = ConditionalOptimizer(loss=fmin_loss_fn, x0=x0, cons=inequality_fn)
        fmin_results = new_ops.solve()

        w, b, i = fmin_results[:2 * self.hidden_dim], fmin_results[2 * self.hidden_dim:], 0
        parameter = [torch.Tensor(np.reshape(w, (2, -1))), torch.Tensor(np.reshape(b, (2, -1)))]
        for params, params_update in zip(model.parameters(), parameter):
            if i == 1:
                params.data.copy_(params_update.squeeze())
            else:
                params.data.copy_(params_update)
            i += 1

        loss_con, loss_per_data_point_con, accuracy_con = test_con(train_loader, model, correct_prediction,
                                                                   loss_per_data_point_con)
        for iter in range(self.max_data_point):

            print("Start iter=%d------------------------------------" % iter)

            loss_per_data_point_diff = loss_per_data_point_uncon - loss_per_data_point_con
            perturb_rank = np.argsort(np.abs(loss_per_data_point_diff))
            l = len(perturb_rank)

            if not plot:
                removed_data_x.append(images_train[perturb_rank[l - 1]])
                removed_data_y.append(cls_train[perturb_rank[l - 1]])
                images_train = np.delete(images_train, perturb_rank[l - 1], axis=0)
            else:
                sample = np.array([X_train[perturb_rank[l - 1]]])
                label = np.array([cls_train[perturb_rank[l - 1]]])

            X_train_new = np.delete(X_train, perturb_rank[l - 1], axis=0)
            y_train_new = np.delete(y_train, perturb_rank[l - 1], axis=0)
            cls_train_new = np.delete(cls_train, perturb_rank[l - 1], axis=0)

            train_dataset = InputDataset.InputDataset(inputs=X_train_new, results=cls_train_new, label=y_train_new,
                                                      input_dim=len(X_train_new))
            train_loader = DataLoader(train_dataset, batch_size=len(X_train_new), num_workers=4, shuffle=False)

            if plot:
                contour_data = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
                logits = model(contour_data)
                m = nn.Softmax()
                Z = m(logits)[:, 1].detach().numpy()
                viz(score=accuracy_con, Z=Z, iter=iter, X_train=X_train_new, X_test=test_point_x.numpy(),
                    y_train=cls_train_new,
                    y_test=np.array([cls_test[test_idx]]), xx=xx, yy=yy, sample=sample, label=label, input_x=X_train,
                    input_y=cls_train)

            X_train, y_train, cls_train = X_train_new, y_train_new, cls_train_new

            print('test point 1')
            for params in model.parameters():
                print(params)

            for epoch_idx in range(1, self.num_epochs + 1):
                train_loss, loss_per_data_point_uncon, accuracy = minimize_(train_loader, model, optimizer, epoch_idx,
                                                                            loss_per_data_point_uncon, correct_prediction)
                if epoch_idx % 50 == 0:
                    print('Epoch [%d]: Train loss: %.3f Accuracy: %.3f' % (epoch_idx, train_loss, accuracy))

            print('test point 2')
            for params in model.parameters():
                print(params)

            train_loss, loss_per_data_point_uncon, accuracy = test_con(train_loader, model, correct_prediction, loss_per_data_point_uncon)
            print("Unconstrained #{}, training acc {}, loss {} ".format(iter, accuracy, train_loss))

            test_loss, test_accuracy = test(test_loader, model, correct_prediction_test)
            print("Unconstrained #{}, testing acc {}, loss {} ".format(iter, test_accuracy, test_loss))

            # inequality keep changing here
            test_point_logit = model(test_point_x)
            m = nn.Softmax()
            test_point_softmax = m(test_point_logit)
            softmax_result = test_point_softmax.detach().numpy()[0]

            if softmax_result[idx] > softmax_result[idx_f]:
                category_idx.append(idx)
            else:
                category_idx.append(idx_f)

            probability.append("%.2f" % round(softmax_result[idx], 2))

            # fmin_loss_fn = self.get_train_fmin_loss_fn(model, train_loader)
            # inequality_fn = self.get_inequality_fn(test_point_logit, idx, idx_f, self.epsilon, model)
            # callback_fn = self.get_callback_fn(model.parameters())
            # x0 = np.random.uniform(-0.5, 0.5, 2 * (self.hidden_dim + 1))

            # inequalities = [- ((test_point_logit[0][idx] - test_point_logit[0][idx_f]) + self.epsilon)]
            print("Probability before constraint: ", softmax_result[idx])

            params_array = np.zeros(2 * (self.hidden_dim + 1))
            for idx, params in enumerate(model.parameters()):
                if idx == 0:
                    params_array[: 2 * self.hidden_dim] = params.detach().numpy().reshape(-1)
                else:
                    params_array[2 * self.hidden_dim:] = params.detach().numpy()

            fmin_results = new_ops.solve()
            # fmin_results = fmin_slsqp(
            #     func=fmin_loss_fn,
            #     x0=x0,
            #     f_ieqcons=inequality_fn,
            #     acc=1e-4)
            # fmin_results = minimize(fun=fmin_loss_fn, x0=x0, method='SLSQP', constraints={'type': 'ineq', 'fun': inequality_fn})['x']
            # pdb.set_trace()
            print("Parameter mean square error: ", mean_squared_error(params_array, fmin_results))
            w, b, i = fmin_results[:2 * self.hidden_dim], fmin_results[2 * self.hidden_dim:], 0
            parameter = [torch.Tensor(np.reshape(w, (2, -1))), torch.Tensor(np.reshape(b, (2, -1)))]
            for params, params_update in zip(model.parameters(), parameter):
                if i == 1:
                    params.data.copy_(params_update.squeeze())
                else:
                    params.data.copy_(params_update)
                i += 1

            print('test point 3')
            for params in model.parameters():
                print(params)

            loss_con, loss_per_data_point_con, accuracy_con = test_con(train_loader, model, correct_prediction, loss_per_data_point_con)

            print("Check probability after constrained optimization.")
            test_point_logit = model(test_point_x)
            m = nn.Softmax()
            test_point_softmax = m(test_point_logit)
            softmax_result = test_point_softmax.detach().numpy()[0]
            print("Probability after constraint: ", softmax_result[idx])
            print("Constrained #{}, training acc {}, loss {} ".format(iter, accuracy_con, loss_con))

            if self.epsilon + train_loss >= loss_con:
                print('Confidence change: ', probability)
                break

    def get_train_fmin_loss_fn(self, model, data_loader):

        def fmin_loss(W):

            w, b, i = W[:2 * self.hidden_dim], W[2 * self.hidden_dim:], 0
            parameter = [torch.Tensor(np.reshape(w, (2, -1))), torch.Tensor(np.reshape(b, (2, -1)))]
            for params, params_update in zip(model.parameters(), parameter):
                if i == 1:
                    params.data.copy_(params_update.squeeze())
                else:
                    params.data.copy_(params_update)
                i += 1
            loss_val = computer_optim_loss(model, data_loader).numpy()
            return loss_val

        return fmin_loss

    def get_inequality_fn(self, test_point_softmax, idx, idx_f, epsilon, model):

        def inequality(W):

            w, b, i = W[:2 * self.hidden_dim], W[2 * self.hidden_dim:], 0
            parameter = [torch.Tensor(np.reshape(w, (2, -1))), torch.Tensor(np.reshape(b, (2, -1)))]
            for params, params_update in zip(model.parameters(), parameter):
                if i == 1:
                    params.data.copy_(params_update.squeeze())
                else:
                    params.data.copy_(params_update)
                i += 1

            return [- ((test_point_softmax[0][idx] - test_point_softmax[0][idx_f]) + epsilon)]

        return inequality

    def get_callback_fn(self, original_parameter):

        params_array = np.zeros(2 * (self.hidden_dim + 1))
        for idx, params in enumerate(original_parameter):
            if idx == 0:
                params_array[: 2 * self.hidden_dim] = params.detach().numpy().reshape(-1)
            else:
                params_array[2 * self.hidden_dim:] = params.detach().numpy()

        def callback_fn(W):

            print("Parameter mean square error: ", mean_squared_error(params_array, W))

        return callback_fn

#
# # TODO: Change the directory.
# transfer_learning.plot_images(path='/Users/hanxing/Desktop', images=removed_data_x, cls_true=removed_data_y,
#                               test_image=images_test[test_idx], test_label=cls_test[test_idx], category_idx=category_idx,
#                               probability=probability)
