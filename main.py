# This file is a modified version of the main.py file from the COMPSYS 302 labs. It has been modified to train and
# test the der_CNN and der_RNN models we have derived from the lab code as a part of this Python project.

from __future__ import print_function
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.der_cnn import der_CNN
from models.der_rnn import der_RNN
import torch.nn.functional as func
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import metrics


# This function trains the der_CNN model.
def train_der_cnn(log_interval, model, device, train_loader, optimizer, epoch, losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Clear the parameter gradients to prevent accumulation of existing gradients
        optimizer.zero_grad()
        output = model(data)  # Forward
        loss = func.nll_loss(output, target)
        losses.append(loss)  # Collect the losses to be averaged for each epoch for plotting.
        loss.backward()  # Backward
        optimizer.step()  # Optimize (carry out the updates)
        # Print training log
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# This function trains the der_RNN model.
def train_der_rnn(log_interval, model, device, train_loader, optimizer, epoch, losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Clear the parameter gradients to prevent accumulation of existing gradients
        optimizer.zero_grad()
        # Reset hidden states
        model.hidden = model.init_hidden()
        data = data.view(-1, 28, 28)
        outputs = model(data)  # Forward
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, target)
        losses.append(loss)  # Collect the losses to be averaged for each epoch for plotting.
        loss.backward()  # Backward
        optimizer.step()  # Optimize (carry out the updates)
        # Print training log
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# This function tests any of the given models.
def test(model, device, test_loader, model_bool, initial_loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # For the example der_CNN model, the data size is (1000, 1, 28, 28). For the der_RNN model, we only need
            # a data size of (1000, 28, 28). The extra dimension (1) needs to be squeezed out when working with the
            # der_RNN model.
            if model_bool:
                data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += func.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    initial_loss.append(test_loss / 10000)
    return 100. * correct / len(test_loader.dataset)


def main():
    # Model selection (if dRNN = False, der_CNN is utilised, and vice versa)
    dRNN = False

    # Training and testing specifications
    if dRNN:
        epochs = 14
    else:
        epochs = 8
    gamma = 0.7
    log_interval = 10
    torch.manual_seed(1)
    l_rate = 0.01
    save_model = True
    load_model = False

    # Check whether the current machine can utilize Cuda to speed up training, and use it if possible.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    ########################### Torchvision ###########################
    # Load MNIST dataset.
    # Use data predefined loader
    # Pre-processing by using the transform.Compose
    # divide into batches
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=1000, shuffle=True, **kwargs)

    ########################### Build Model ###########################
    if dRNN:
        model = der_RNN(device).to(device)
    else:
        model = der_CNN().to(device)

    # Set optimizer
    if dRNN:
        optimizer = optim.Adadelta(model.parameters(), lr=l_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=l_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    ########################### Training and Testing ###########################
    train_losses = []
    accuracies = []
    temp_list = []
    # Test with initialised network parameters to see pre-training results.
    accuracy = test(model, device, test_loader, dRNN, temp_list)
    initial_loss = temp_list[0]     # Store the pre-training loss for plotting.
    temp_list = []
    accuracies.append(accuracy)     # Store the pre-training accuracy for plotting.
    # Train model, testing it after each epoch.
    for epoch in range(1, epochs + 1):
        if dRNN:
            train_der_rnn(log_interval, model, device, train_loader, optimizer, epoch, temp_list)
        else:
            train_der_cnn(log_interval, model, device, train_loader, optimizer, epoch, temp_list)
        # Average the training losses for each epoch for plotting.
        train_losses.append(sum(temp_list) / len(temp_list))
        temp_list = []
        print('Testing...')
        accuracy = test(model, device, test_loader, dRNN, temp_list)
        temp_list = []
        accuracies.append(accuracy)  # Collect the accuracy when tested after each epoch for plotting.
        scheduler.step()
    # Prepend pre-training loss for plotting training losses.
    train_losses.insert(0, initial_loss)

    ########################### Saving & Loading Model ###########################
    if save_model:
        if dRNN:
            torch.save(model.state_dict(), "./results/der_RNN.pt")
        else:
            torch.save(model.state_dict(), "./results/der_CNN.pt")
    if load_model:
        if dRNN:
            torch.load(model.state_dict(), "./results/der_RNN.pt")
        else:
            torch.load(model.state_dict(), "./results/der_CNN.pt")

    ########################### Graph Generation ###########################
    plt.figure()
    # Average training loss for each epoch.
    plt.subplot(2, 1, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Training Loss')
    # Accuracy after each epoch.
    plt.subplot(2, 1, 2)
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # Save and show figure.
    if dRNN:
        plt.savefig('./results/der_RNN results.png')
    else:
        plt.savefig('./results/der_CNN results.png')
    plt.show()

    print('Graphs have been saved.')

    # Confusion matrix, and precision and recall (among other metrics)
    # print(metrics.confusion_matrix(labels_true, labels_pred))
    # print(metrics.classification_report(labels_true, labels_pred, digits = 10))


if __name__ == '__main__':
    main()
