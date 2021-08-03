import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from BaseModel import BaseModel
from SecondModel import SecondModel


baseModelDevice = 'cuda:0' if torch.cuda.is_available() else 'cpu'
secondModelDevice = 'cuda:1' if torch.cuda.is_available() else 'cpu'


def train_cifar10():
    train_dataset = datasets.CIFAR10(root='data',
                                     download=True,
                                     train=True,
                                     transform=ToTensor()
                                     )
    test_dataset = datasets.CIFAR10(root='data',
                                    download=True,
                                    train=False,
                                    transform=ToTensor()
                                    )

    train_data_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    model = BaseModel().to(baseModelDevice)
    second_model = SecondModel().to(secondModelDevice)
    print('-----------------Model:1----------------')
    print(model)
    print('----------------------------------------')
    print('-----------------Model:2----------------')
    print(second_model)
    print('----------------------------------------')
    base_loss_fn = nn.CrossEntropyLoss().to(baseModelDevice)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    sec_loss_fn = nn.CrossEntropyLoss().to(secondModelDevice)
    sec_optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    base_model_epoch_vs_mean_loss = []
    sec_model_epoch_vs_mean_loss = []
    base_model_test_acc = []
    sec_model_test_acc = []
    epoch = 10
    for i in range(epoch):
        print(f'Epoch:{i}')
        losses = train(train_data_loader, model, base_optimizer, base_loss_fn, second_model, sec_optimizer, sec_loss_fn)
        base_model_epoch_vs_mean_loss.append(losses[0])
        sec_model_epoch_vs_mean_loss.append(losses[1])

        test_acc = test(test_data_loader, model, second_model)
        base_model_test_acc.append(test_acc[0])
        sec_model_test_acc.append(test_acc[1])
    plot(epoch, base_model_epoch_vs_mean_loss, sec_model_epoch_vs_mean_loss, base_model_test_acc, sec_model_test_acc)


def plot(epoch, base_loss, sec_model_loss, base_test_acc, sec_test_acc):
    x_axis = np.arange(epoch)
    fig = plt.figure(figsize=[8, 4], facecolor='white', constrained_layout=True)
    gs = fig.add_gridspec(4, 4)

    plt.rc('font', size=3)
    plt.rcParams['font.size'] = 5
    plt.rcParams['text.color'] = '#8d8c8c'

    axs = fig.add_subplot(gs[1:3, :2])
    plot_sub_graph(axs, x_axis, base_loss, sec_model_loss, 'Average Loss value', "Base model training loss",
                   "Second model training loss")
    axs = fig.add_subplot(gs[1:3, 2:])
    plot_sub_graph(axs, x_axis, base_test_acc, sec_test_acc, 'Test Accuracy', 'Base model test accuracy',
                   'Second model test accuracy')
    plt.savefig("./model_comparison.png", dpi=300)
    plt.close()


def plot_sub_graph(axs, x_axis, y_axis_1, y_axis_2, y_label, legend1, legend2):
    axs.spines['top'].set_color('#c4c4c4')
    axs.spines['left'].set_color('#c4c4c4')
    axs.spines['bottom'].set_color('#c4c4c4')
    axs.spines['right'].set_color('#c4c4c4')
    for label in (axs.get_xticklabels() + axs.get_yticklabels()):
        label.set_fontsize(2)
    axs.set_xticks(x_axis, minor=True)
    axs.xaxis.grid(True, linewidth=0.3, color='#eeeded', linestyle='-')
    axs.yaxis.grid(True, linewidth=0.60, color='#eeeded', linestyle='-')
    axs.tick_params(
        axis='x', labelsize=5, length=0, width=0,
        labelcolor='#8d8c8c'
    )
    axs.tick_params(
        axis='y', labelsize=5, length=0, width=0,
        labelcolor='#8d8c8c'
    )

    axs.plot(x_axis, y_axis_1, marker='o', markerfacecolor='green', markersize=0.5, color='green', linewidth=0.25,
             label=legend1)
    axs.plot(x_axis, y_axis_2, marker='o', markerfacecolor='red', markersize=0.5, color='red', linewidth=0.25,
             label=legend2)
    axs.set_xlabel('Epoch')
    axs.set_ylabel(y_label)

    axs.legend(loc='center left', frameon=False, facecolor='white')


def train(data_loader, model, optimizer, loss_fn, sec_model, sec_optim, sec_loss_fn):
    base_model_loss = []
    sec_model_loss = []
    for batch, (X, y) in enumerate(data_loader):
        X_base, y_base = X.to(baseModelDevice), y.to(baseModelDevice)
        X_sec, y_sec = X.to(secondModelDevice), y.to(secondModelDevice)
        output_ = model(X_base)
        output_sec_model = sec_model(X_sec)

        loss = loss_fn(output_, y_base)
        base_model_loss.append(loss)
        loss_sec = sec_loss_fn(output_sec_model, y_sec)
        sec_model_loss.append(loss_sec)

        optimizer.zero_grad()
        sec_optim.zero_grad()

        loss.backward()
        loss_sec.backward()

        optimizer.step()
        sec_optim.step()

        if batch % 100 == 0:
            print(f"Base model Loss: {loss}")
            print(f"Second model Loss: {loss_sec}")

    return [np.array(base_model_loss, dtype=np.float).mean(),
            np.array(sec_model_loss, dtype=np.float).mean()]


def test(data_loader, model, sec_model):
    n_correct = 0
    sec_n_correct = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(baseModelDevice), y.to(baseModelDevice)
            X_sec, y_sec = X.to(secondModelDevice), y.to(secondModelDevice)
            output_ = model(X)
            output_sec = sec_model(X_sec)
            n_correct += (output_.argmax(1) == y).type(torch.float).sum().item()
            sec_n_correct += (output_sec.argmax(1) == y_sec).type(torch.float).sum().item()
        n_correct = n_correct / len(data_loader.dataset)
        sec_n_correct = sec_n_correct / len(data_loader.dataset)
    print(f'Base model Test Accuracy:{n_correct}')
    print(f'Second model Test Accuracy:{sec_n_correct}')
    return [n_correct, sec_n_correct]


if __name__ == '__main__':
    train_cifar10()
