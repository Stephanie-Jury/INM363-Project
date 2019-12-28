import torch
import pandas as pd
import time
import matplotlib
import torch.optim as optim
from adabound import AdaBound

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Source: https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch/blob/master/model/capsules.py
class AverageMeter(object):
    """Computes and stores the average from a running total"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Source: https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch/blob/master/model/capsules.py
def accuracy(output, target, topk=(1,)):
    """Computes accuracy for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred).type(torch.cuda.LongTensor))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Source: https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch/blob/master/model/capsules.py
def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    train_len = len(train_loader)

    epoch_loss = 0
    epoch_acc = 0
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        r = (1. * batch_idx + (epoch - 1) * train_len) / (32 * train_len)
        loss = criterion(output, target, r)
        acc = accuracy(output, target)
        loss.backward()

        # Perform a parameter update based on the current gradient (.grad attribute) and the update rule
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        epoch_loss += loss.item()
        epoch_acc += acc[0].item()

        if batch_idx:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Accuracy: {:.6f}\tLoss: {:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                acc[0].item(),
                loss.item(),
                batch_time=batch_time,
                data_time=data_time)
            )

    epoch_acc /= train_len
    epoch_loss /= train_len

    return epoch_acc, epoch_loss

# Source https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch/blob/master/train.py
def test(data_loader, model, criterion, phase, device):
    model.eval()
    loss = 0
    acc = 0
    test_len = len(data_loader)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target, r=1).item()
            acc += accuracy(output, target)[0].item()

    loss /= test_len
    acc /= test_len

    print('\n{} set average accuracy: {:.3f}, loss: {:.3f} \n'.format(phase, acc, loss))
    return acc, loss

# Source https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch/blob/master/train.py
def generate_plots(file_name, type, out_name):
    df = pd.read_csv(file_name)

    df_train = df[df['Phase'] == 'Training']
    df_dev = df[df['Phase'] == 'Validation']

    plt.xlabel('Epoch Number')
    plt.ylabel(type)

    plt.plot(df_train['Epoch'], df_train[type], label='Training')
    plt.plot(df_dev['Epoch'], df_dev[type], label='Validation')
    plt.legend()

    plt.savefig(out_name)


def save_model_info(model, output_path, epoch):
    path = os.path.join(output_path, 'model_{}.pth'.format(epoch))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print('Saving model to {}'.format(path))
    torch.save(model.state_dict(), path)


def load_optimiser(model, optimiser_selection):
    return {'adam': optim.Adam(model.parameters(), lr=0.01, weight_decay=0),
            'adagrad': optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0),
            'adabound': AdaBound(model.parameters(), lr=0.01, final_lr=0.1),
            'SGD': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
            }[optimiser_selection]

def load_scheduler(optimiser, scheduler_selection):
    return {'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'max', patience=1),
            'StepLR': optim.lr_scheduler.StepLR(optimiser, 20, gamma=0.8)
            }[scheduler_selection]


def model_train_and_test(checkpoint_frequency, epochs, train_loader, val_loader, test_loader, model, scheduler_selection, criterion,
                         optimiser_selection, device, output_path):

    optimiser = load_optimiser(model, optimiser_selection)
    scheduler = load_scheduler(optimiser, scheduler_selection)

    completed_epoch_count = 0

    accuracy_csv = open(output_path + "/Accuracy.csv", "w")
    loss_csv = open(output_path + "/Loss.csv", "w")

    accuracy_csv.write('Epoch,Phase,Accuracy\n')
    loss_csv.write('Epoch,Phase,Loss\n')

    for checkpoint in range(1, int(1 / checkpoint_frequency) + 1):

        for epoch in range(1, epochs + 1):

            torch.cuda.empty_cache()

            print('Epoch {}/{}'.format(epoch, epochs))
            print('-' * 30)

            train_acc, train_loss = train(train_loader, model, criterion, optimiser, epoch, device)

            accuracy_csv.write('{},{},{:.3f}\n'.format(epoch, 'Training', train_acc))
            loss_csv.write('{},{},{:.3f}\n'.format(epoch, 'Training', train_loss))

            dev_acc, dev_loss = test(val_loader, model, criterion, 'Validation', device)

            accuracy_csv.write('{},{},{:.3f}\n'.format(epoch, 'Validation', dev_acc))
            loss_csv.write('{},{},{:.3f}\n'.format(epoch, 'Validation', dev_loss))

            scheduler.step(train_acc)

            # Print console output
            print('Checkpoint: ' + str(checkpoint) + ' of ' + str(int(1 / checkpoint_frequency)))

    # Save accuracy and loss data and generate plots
    accuracy_csv.close()
    loss_csv.close()

    generate_plots(output_path + "/Accuracy.csv", 'Accuracy', output_path + "/Accuracy.png", epochs)
    generate_plots(output_path + "/Loss.csv", 'Loss', output_path + "/Loss.png", epochs)

    test_acc, test_loss = test(test_loader, model, criterion, 'Test', device)

    out_test = open(output_path + "/test.txt", "w")
    out_test.write(
        'Epoch: {}, Accuracy: {:.3f}, Loss: {:.3f} \n'.format(completed_epoch_count, test_acc, test_loss))
    out_test.close()

    save_model_info(model, output_path, completed_epoch_count)
