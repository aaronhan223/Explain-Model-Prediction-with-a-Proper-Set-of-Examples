import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import pdb

num_epochs = 100


def train(data_loader, model, optimizer, epoch, embed):
    model.train()
    total_loss = 0.0

    for batch_idx, batch_data in enumerate(data_loader):
        inputs = Variable(batch_data['InputVector'].float())
        results = Variable(batch_data['Result'].long())

        optimizer.zero_grad()
        predictions, representation = model(inputs)
        predictions = torch.squeeze(predictions)
        loss = F.cross_entropy(predictions, results)
        loss.backward()
        optimizer.step()

        total_loss += loss.data
        representation = representation.detach().numpy()
        embed = representation

    return total_loss, embed


def minimize(data_loader, model, optimizer, epoch, individual_loss, correct_prediction):

    model.train()
    total_loss = 0.0

    for batch_idx, batch_data in enumerate(data_loader):
        inputs = Variable(batch_data['InputVector'].float())
        results = Variable(batch_data['Result'].long())
        label = Variable(batch_data['Label'].float())

        optimizer.zero_grad()
        predictions = model(inputs)
        predictions = torch.squeeze(predictions)
        loss = F.cross_entropy(predictions, results)
        loss.backward()
        optimizer.step()
        correct_prediction = torch.eq(torch.argmax(predictions, dim=1), torch.argmax(label, dim=1)).float().detach().numpy()
        total_loss += loss.data

        loss_per_data = torch.sum(- label * F.log_softmax(predictions, -1), -1)
        loss_per_data = loss_per_data.detach().numpy()
        individual_loss = loss_per_data

    accuracy = np.mean(correct_prediction)

    return total_loss, individual_loss, accuracy


def test_con(data_loader, model, correct_prediction, individual_loss):

    model.eval()
    total_loss = 0.0

    for batch_idx, batch_data in enumerate(data_loader):
        inputs = Variable(batch_data['InputVector'].float())
        results = Variable(batch_data['Result'].long())
        label = Variable(batch_data['Label'].float())

        predictions = model(inputs)
        predictions = torch.squeeze(predictions)
        loss = F.cross_entropy(predictions, results)
        loss_per_data = torch.sum(- label * F.log_softmax(predictions, -1), -1)
        correct_prediction = torch.eq(torch.argmax(predictions, dim=1), torch.argmax(label, dim=1)).float().detach().numpy()
        total_loss += loss.data
        loss_per_data = loss_per_data.detach().numpy()
        individual_loss = loss_per_data

    accuracy = np.mean(correct_prediction)

    return total_loss, individual_loss, accuracy


def test(data_loader, model, correct_prediction):

    model.eval()
    total_loss = 0.0

    for batch_idx, batch_data in enumerate(data_loader):
        inputs = Variable(batch_data['InputVector'].float())
        results = Variable(batch_data['Result'].long())
        label = Variable(batch_data['Label'].float())

        predictions = model(inputs)
        predictions = torch.squeeze(predictions)
        loss = F.cross_entropy(predictions, results)
        correct_prediction = torch.eq(torch.argmax(predictions, dim=1), torch.argmax(label, dim=1)).float().detach().numpy()
        total_loss += loss.data

    accuracy = np.mean(correct_prediction)

    return total_loss, accuracy


def computer_optim_loss(model, data_loader):

    model.eval()
    total_loss = 0.0

    for batch_idx, batch_data in enumerate(data_loader):
        inputs = Variable(batch_data['InputVector'].float())
        results = Variable(batch_data['Result'].long())

        predictions = model(inputs)
        predictions = torch.squeeze(predictions)
        loss = F.cross_entropy(predictions, results)
        total_loss += loss.data

    return total_loss
