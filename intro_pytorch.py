from os import times
import numpy as np
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training=True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('./data', train=True, download=True,
                               transform=custom_transform)
    test_set = datasets.MNIST('./data', train=False,
                              transform=custom_transform)
    if training == False:
        loader = torch.utils.data.DataLoader(
            test_set, batch_size=50, shuffle=False)
        return loader
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=50, shuffle=True)
    return loader


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    for epoch in range(T):
        train_loss = 0.0
        total_true = 0
        total_iter = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            predic = torch.argmax(outputs, 1)
            total_true += (predic == labels).sum().item()
            total_iter += 1

        print("Train Epoch: %d 	 Accuracy: %5d/60000(%.2f%%) Loss: %.3f" % (epoch,
              total_true, total_true/600, train_loss / len(train_loader)))


def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.eval()
    test_loss = 0.0
    total_true = 0
    total_iter = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data

            # optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()

            test_loss += loss.item()

            predic = torch.argmax(outputs, 1)
            total_true += (predic == labels).sum().item()
            total_iter += 1
    if show_loss:
        print("Average loss: %.4f" %
              (test_loss / (len(test_loader) * test_loader.batch_size)))
    print("Accuracy: %.2f%%" % (total_true/100))


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['zero', 'one', 'two', 'three',
                   'four', 'five', 'six', 'seven', 'eight', 'nine']
    model.eval()

    outputs = model(test_images[index])
    prob = F.softmax(outputs, dim=1)
    for iter in range(3):
        i = torch.argmax(prob).item()
        val = torch.max(prob).item()
        print(class_names[i], ': ', "%.2f%%" % (val*100), sep='')
        prob[0][i] = 0

    # print(prob[leng-1])
    # print(prob[leng-2])
    # print(prob[leng-3])

    # print("Prob:", prob)
    # print("Prob Type:", type(prob))
    # print("Prob Len:", len(prob))
    # print("Prob[0] Len:", len(prob[0]))
    # print("Outputs:", outputs)
    # print("Outputs Type:", type(outputs))
    # print("Outputs Len:", len(outputs))

    # return outputs, prob


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
