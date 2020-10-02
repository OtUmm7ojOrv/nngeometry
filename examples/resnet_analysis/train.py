import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import time

start_time = time.time()

# dataset
means = (0.4802, 0.4481, 0.3975)
normalize = transforms.Normalize(mean=means,
                                 std=[0.2733, 0.2658, 0.2777])
tr_train = transforms.Compose([transforms.RandomCrop(64, padding=6, padding_mode='reflect'),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               normalize])
tr_test = transforms.Compose([transforms.ToTensor(),
                              normalize])

# 100.000 (500 x 200) examples in train set
trainset = datasets.ImageFolder('/tmp/data/tiny-imagenet-200/train/', transform=tr_train)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
testset = datasets.ImageFolder('/tmp/data/tiny-imagenet-200/val/', transform=tr_test)
testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)

# log

columns = ['iteration', 'time',
           'train_loss', 'train_acc',
           'test_loss', 'test_acc',
           'epoch']

log = pd.DataFrame(columns=columns)

# model

from resnet import resnet50
# from resnet import resnet50
model = resnet50(num_classes=200).to('cuda')
model.train()

# optimizer

base_lr = .1
optimizer = optim.SGD(model.parameters(),
                      lr=base_lr,
                      momentum=.9,
                      weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()

def adjust_learning_rate(optimizer, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # with bs = 64, 1 epoch ~ 1500 sgd iterations
    lr = base_lr * (0.1 ** (iteration // (1500 * 30)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('lr set to ' + str(lr))

# test

def test():
    loss = 0
    acc = 0
    n = 0
    with torch.no_grad():
        for x, y in iter(testloader):
            x = x.to('cuda')
            y = y.to('cuda')
            out = model(x)
            loss += criterion(out, y).item()
            _, pred = out.max(1)
            acc += (pred.eq(y).sum().float() / x.size(0)).item()
            n += 1
    acc /= n
    loss /= n
    return loss, acc

# running average estimates of current train loss and accuracy

loss_ra = 7
acc_ra = 0
gamma = .1
next_percent = 2

# train

iteration = 0
for epoch in range(120):
    for x, y in iter(trainloader):
        optimizer.zero_grad()
        x = x.to('cuda')
        y = y.to('cuda')

        out = model(x)
        loss = criterion(out, y)
        _, pred = out.max(1)
        acc = pred.eq(y).sum().float() / x.size(0)

        loss_ra = gamma * loss.item() + (1 - gamma) * loss_ra
        acc_ra = gamma * acc.item() + (1 - gamma) * acc_ra

        loss.backward()
        optimizer.step()

        # if iteration % 5 == 0 or acc_ra*100 > next_percent:
        if iteration % 1500 == 0 or acc_ra*100 > next_percent:
            test_loss, test_acc = test()

            to_log = pd.Series()
            to_log['time'] = time.time() - start_time
            to_log['iteration'] = iteration
            to_log['epoch'] = epoch
            to_log['train_loss'] = loss_ra
            to_log['test_loss'] = test_loss
            to_log['train_acc'] = acc_ra
            to_log['test_acc'] = test_acc

            log.loc[len(log)] = to_log
            print(log.loc[len(log) - 1])

            log.to_pickle('log.pkl')

            if acc_ra*100 > next_percent:
                torch.save(model, 'saved_model/%d.pth.tar' % iteration)
                next_percent += 2
            elif iteration % (1500 * 10) == 0:
                torch.save(model, 'saved_model/%d.pth.tar' % iteration)

            adjust_learning_rate(optimizer, iteration)

        iteration += 1