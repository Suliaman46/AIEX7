import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import optim
from matplotlib import pyplot
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        inp, HL, out = [1, 16, 1]
        self.linear1 = nn.Linear(inp, HL)
        self.linear2 = nn.Linear(HL, out)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        return x


class FuncDataset(Dataset):
    def __init__(self, x, y):
        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor
        self.length = x.shape[0]
        self.x_data = torch.from_numpy(x).type(x_dtype)
        self.y_data = torch.from_numpy(y).type(y_dtype)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


def train_batch(model, x, y, optimizer, loss_fn):
    # Running forward calculation
    y_predict = model.forward(x)
    # Compute loss.
    loss = loss_fn(y_predict, y)
    # Zeroing all the gradients for all parameters
    optimizer.zero_grad()
    # Backward pass: Calculating the gradients for each parameter
    loss.backward()
    # Adjusting each parameter by its calculated gradient
    optimizer.step()
    return loss.data.item()


def train(model, loader, optimizer, loss_fn, epochs=5):
    losses = list()
    batch_index = 0
    for e in range(epochs):
        for x, y in loader:
            loss = train_batch(model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn)
            losses.append(loss)
            batch_index += 1
        print("Epoch: ", e + 1)
        print("Batches: ", batch_index)
    return losses

def test_batch(model, x, y):
    # run forward calculation
    y_predict = model.forward(x)
    # returning prediction as well as target value
    return y, y_predict

def test(model, loader):
    y_vectors = list()
    y_predict_vectors = list()

    batch_index = 0
    for x, y in loader:
        y, y_predict = test_batch(model=model, x=x, y=y)

        y_vectors.append(y.data.numpy())
        y_predict_vectors.append(y_predict.data.numpy())

        batch_index += 1

    y_predict_vector = np.concatenate(y_predict_vectors)

    return y_predict_vector


def run(dataset_train, dataset_test, learning_rate):
    batch_size_train = 16     # number of training examples used to calculate each iteration's gradient
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), shuffle=False)
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.1)
    loss_fn = nn.L1Loss()  # Least Absolute Deviations

    loss = train(model=model, loader=data_loader_train, optimizer=optimizer, loss_fn=loss_fn)
    prediction = test(model=model, loader=data_loader_test)

    return loss, prediction

def running_mean(x,N):
    # Running mean is equivalent to convolving target vector x with an N long array with each member equal to 1/N
    return np.convolve(x, np.ones(N), 'valid') / N

def learning_curve(losses):
    loss_mean = running_mean(losses,400)
    fig,ax = pyplot.subplots()
    x_axes1 = list(range(len(losses)))
    x_axes2 = list(range(len(loss_mean)))
    data_line = ax.plot(x_axes1,losses)
    mean_line = ax.plot(x_axes2, loss_mean)
    pyplot.xlabel('Iterations')
    pyplot.ylabel('Loss')

# Generating our data
train_sample_size = 200000
test_sample_size = 10000
x_train = np.random.uniform(-10, 10, [train_sample_size, 1])
y_train = (np.sin(x_train * np.sqrt(1)) + np.cos(x_train * np.sqrt(2)))
y_train += (np.random.randn(train_sample_size, 1)/25)  # Adding Noise

x_test = np.random.uniform(-10, 10, [test_sample_size, 1])
y_test = (np.sin(x_test * np.sqrt(1)) + np.cos(x_test * np.sqrt(2)))
y_temp = y_test.copy()

# # Whitening Input Data
x_train_whitened = (x_train - x_train.mean())/x_train.std()
x_test_whitened = (x_test - x_test.mean())/x_test.std()

dataset_train = FuncDataset(x=x_train_whitened, y=y_train)
dataset_test = FuncDataset(x=x_test_whitened, y=y_test)

# #Code to run the model without the whitening transformation
# dataset_train = FuncDataset(x=x_train, y=y_train)
# dataset_test = FuncDataset(x=x_test, y=y_test)

print("Train set size: ", dataset_train.length)
print("Test set size: ", dataset_test.length)

#Plotting Train Data
pyplot.figure(1)
pyplot.title('Training Data with noise')
pyplot.scatter(x_train, y_train, marker='o', s=0.2)

#Running the Training as well as testing
learning_rate = 0.049
losses, y_predict = run(dataset_train=dataset_train, dataset_test=dataset_test, learning_rate=learning_rate)

#Plotting Learning curve
learning_curve(losses)

y_predict = y_predict

#Plotting Final Results
pyplot.figure(3)
pyplot.title('Target Function Plot vs Prediction')
pyplot.scatter(x_test, y_temp, marker='o',s = 0.5)
pyplot.scatter(x_test, y_predict, marker='o',s = 0.5)
pyplot.text(-9, 0.45, "- Prediction", color="orange", fontsize=10)
pyplot.text(-9, 0.55, "- Function", color="blue", fontsize=10)
pyplot.text(1.5, 1.8, "No. Nodes: " + str(16))
pyplot.text(1.5, 2, "Learning Rate: " + str(learning_rate))
pyplot.show()
