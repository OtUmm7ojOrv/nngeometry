import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import time
import numpy as np
import pickle as pkl

start_time = time.time()

n_samples = 500
reguls = 10. ** np.arange(-4, -.5, .25)

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
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
testset = datasets.ImageFolder('/tmp/data/tiny-imagenet-200/val/', transform=tr_test)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
len(trainset), len(testset)

smalltestdata = next(iter(DataLoader(testset, batch_size=n_samples, shuffle=True, num_workers=4)))
smalltestset = TensorDataset(smalltestdata[0].to('cuda'), smalltestdata[1].to('cuda'))
smalltestloader = DataLoader(smalltestset, batch_size=100, shuffle=False)

from nngeometry.generator import Jacobian
from nngeometry.object import (FMatDense, PMatDense, PMatDiag, PMatImplicit,
                               PMatQuasiDiag, PMatKFAC, PMatEKFAC)
from nngeometry.metrics import FIM, FIM_MonteCarlo

from nngeometry.object.vector import random_pvector
import numpy as np

log = pd.read_pickle('log.pkl')

results = dict()

def save():
    with open('analysis_results.pkl', 'wb') as f:
        pkl.dump(results, f)

models = [0]
for t in [.1, .2, .3, .4, .5]:
    models.append(int(log[(log['test_acc'] > t) &
                          (log['iteration'] % 1500 != 0)].iloc[0]['iteration']))

for model in models:
    m = torch.load('saved_model/%d.pth.tar' % model)
    results[model] = []

    for i in range(5):

        results[model].append([])

        F = FIM_MonteCarlo(m, smalltestloader, PMatImplicit, trials=5,
                           device='cuda')
        # G = FMatDense(F.generator)
        # frob = torch.norm(G.sum(dim=(0, 2))) / n_samples
        tr = F.trace()

        for j in range(4):

            results[model][-1].append(dict())
            v = random_pvector(F.generator.layer_collection, device='cuda')
            v_flat = v.get_flat_representation()
            Fv = F.mv(v)
            Fv_flat = Fv.get_flat_representation()
            vTMv = F.vTMv(v)

            for repr in [PMatDiag, PMatQuasiDiag, PMatKFAC, PMatEKFAC]:
                results[model][-1][-1][repr] = dict()
                F2 = repr(F.generator)
                if repr == PMatEKFAC:
                    F2.update_diag()

                tr_repr = F2.trace()
                results[model][-1][-1][repr]['trace'] = torch.abs((tr_repr - tr) / tr).item()

                Fv_repr = F2.mv(v)
                Fv_repr_flat = Fv_repr.get_flat_representation()
                angle = torch.dot(Fv_flat, Fv_repr_flat) / torch.norm(Fv_flat) / torch.norm(Fv_repr_flat)
                results[model][-1][-1][repr]['Fv'] = angle.item()

                vTMv_repr = F2.vTMv(v)
                results[model][-1][-1][repr]['vTMv'] = torch.abs((vTMv_repr - vTMv) / vTMv).item()

                # frob_repr = F2.frobenius_norm()
                # results[model][-1][-1][repr]['frob'] = torch.abs((frob_repr - frob) / frob).item()

                results[model][-1][-1][repr]['solve'] = []
                for regul in reguls:

                    v_inv = F2.solve(v, regul=regul)
                    v_back = F.mv(v_inv) + regul * v
                    
                    v_back_flat = v_back.get_flat_representation()
                    
                    angle = torch.dot(v_flat, v_back_flat) / torch.norm(v_flat) / torch.norm(v_back_flat)

                    results[model][-1][-1][repr]['solve'].append(angle.item())

                # print(results)
                save()

                print(time.time() - start_time)
