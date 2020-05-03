# This file is a modified version of the main.py file from the COMPSYS 302 labs. It has been modified to train and
# test the der_CNN and der_RNN models we have derived from the lab code as a part of this Python project.

from __future__ import print_function
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.conv import Net
from models.rnn_conv import ImageRNN
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# This function saves an image.
def imsave(img):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    im.save("./results/your_file.jpeg")

# This function trains the der_CNN model.
def train_der_cnn(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward(); optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# This function trains the der_RNN model.
def train_der_rnn(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # reset hidden states
        model.hidden = model.init_hidden()
        data = data.view(-1, 28, 28)
        outputs = model(data)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, target)
        loss.backward(); optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# This function tests any of the given models.
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            print(data.shape)
            # For the example CNN model and the der_CNN model, the data size is (1000, 1, 28, 28). For the example RNN
            # model and the der_RNN model, we only need a data size of (1000, 28, 28). The extra dimension (1) needs to
            # be squeezed out when working with the example RNN model and der_RNN model.
            #data = torch.squeeze(data)
            print(data.shape)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training and testing specifications
    epochs = 14
    gamma = 0.7
    log_interval = 10
    torch.manual_seed(1)
    l_rate = 0.01
    save_model = True

    # Model selection
    der_CNN = False
    der_RNN = False

    # der_RNN model specifications
    N_STEPS = 28
    N_INPUTS = 28
    N_NEURONS = 150
    N_OUTPUTS = 10

    # Check whether the current machine can utilize Cuda to speed up training.
    use_cuda = torch.cuda.is_available()
    # Use Cuda if possible.
    device = torch.device("cuda" if use_cuda else "cpu")


    ######################   Torchvision    ###########################
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

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # img = torchvision.utils.make_grid(images)
    # imsave(img)

    ######################    Build model and run   ############################
    # Build model
    if der_RNN:
        model = der_RNN(64, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS, device).to(device)
    else:
        model = der_CNN().to(device)

    # Set optimizer
    if der_RNN:
        optimizer = optim.Adadelta(model.parameters(), lr=l_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=l_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Training and testing
    for epoch in range(1, epochs + 1):
        if der_RNN:
            train_der_rnn(log_interval, model, device, train_loader, optimizer, epoch)
        else:
            train_der_cnn(log_interval, model, device, train_loader, optimizer, epoch)

        test(model, device, test_loader)
        scheduler.step()

    if save_model:
        if der_RNN:
            torch.save(model.state_dict(), "./results/mnist_der_RNN.pt")
        else:
            torch.save(model.state_dict(), "./results/mnist_der_CNN.pt")
    # Figure out how to load saved models to prevent the need for re-training every time main is run.

    # Add graph generation and comparison of two models' accuracy and loss value.


if __name__ == '__main__':
    main()