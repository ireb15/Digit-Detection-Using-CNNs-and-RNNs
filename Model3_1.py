#Import the required packages 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

from torchvision import datasets
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

#Define constants that will be later used in the training of the model
NUM_EPOCHS = 6
BATCH_SIZE = 10
LEARNING_RATE = 0.01

#Define a transform to convert the data to tensors then normalise the data 
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)

#Get the training data from MNIST
trainset = datasets.MNIST(
    root = './data',
    train = True,
    download = True, 
    transform = transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size = 10,
    shuffle = True
)

#Get the testing data from MNIST
testset = datasets.MNIST(
    root = './data',
    train = False,
    download = True,
    transform = transform
)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size = 10,
    shuffle = False
)

#This following code is used to visualise data
for batch_1 in trainloader:
    batch = batch_1
    break
 
print(batch[0].shape) # as batch[0] contains the image pixels -> tensors
print(batch[1]) # batch[1] contains the labels -> tensors
 
plt.figure(figsize=(12, 8))
for i in range (batch[0].shape[0]):
    plt.subplot(1, 10, i+1)
    plt.axis('off')
    plt.imshow(batch[0][i].reshape(28, 28), cmap='gray')
    plt.title(int(batch[1][i]))
    plt.savefig('digit_mnist.png')
plt.show()

#Here we define the model
#This model is based on the LeNet-5 CNN Model
#This model contains 2 convolution layers, 2 pooling layers, 3 fully connected layers 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,   #First convolution layer
                               kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,  #Second convolution layer
                               kernel_size=5)
        self.fc1 = nn.Linear(in_features=256, out_features=120) #First fully connected layer
        self.fc2 = nn.Linear(in_features=120, out_features=84)  #First fully connected layer    
        self.fc3 = nn.Linear(in_features=84, out_features=10)   #First fully connected layer
 
    #The forward function defines the structure of the nework 
    def forward(self, x):
        x = F.torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
print(net)

#Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

#Check if GPU is available, then use GPU if yes, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
net.to(device)

#Function to calculate accuracy
def calc_acc(loader):
    correct = 0
    total = 0
    for data in loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return ((100*correct)/total) 

#This function is used to train the NN
def train():
    epoch_loss = []
    train_acc = []
    test_acc = []

    for epoch in range(NUM_EPOCHS):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
 
            #Set parameter gradients to zero
            optimizer.zero_grad()
 
            #Forward pass
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            #Propogate the loss backwards
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item()
 
        epoch_loss.append(running_loss/6000)
        train_acc.append(calc_acc(trainloader))
        test_acc.append(calc_acc(testloader))
        print('Epoch: %d of %d, Train Acc: %0.3f, Test Acc: %0.3f, Loss: %0.3f'
              % (epoch+1, NUM_EPOCHS, train_acc[epoch], test_acc[epoch], running_loss/6000))
        
    return epoch_loss, train_acc, test_acc

#Start training the NN
start = time.time()
epoch_loss, train_acc, test_acc = train()
end = time.time()
 
print('%0.2f minutes' %((end - start) / 60)) #Prints the time elapsed for the NN to train and test

#Show the epoch loss output graph
plt.figure()
plt.plot(epoch_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_v_ephoch.png')
plt.show()

#Show the training accuracy output graph
plt.figure()
plt.plot(train_acc)
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.savefig('train_acc.png')
plt.show()

#Show the testing accuracy output graph
plt.figure()
plt.plot(test_acc)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.savefig('test_acc.png')
plt.show()

#The following code is for getting the prediction values and actual values
#With these values we are able to output the confusion matrix, f1 score and accuracy score
def test_label_predictions(model, device, testloader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]

actuals, predictions = test_label_predictions(net, device, testloader)
print('Confusion matrix:')
print(confusion_matrix(actuals, predictions))
print('F1 score: %f' % f1_score(actuals, predictions, average='micro'))
print('Accuracy score: %f' % accuracy_score(actuals, predictions))

print('Classification Report:')
print(classification_report(actuals, predictions, target_names= ['0', '1', '2', '3', '4', '5', '6', '7','8','9'] ))