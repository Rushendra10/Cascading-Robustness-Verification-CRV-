import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append(".")
import numpy as np

from convex_adversarial.utils import robust_loss

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self


def evaluate_robust(loader, model, epsilon, epoch, log, verbose):
    model.eval()
    robust = []
    non_robust = []

    for i, (X, y) in enumerate(loader):
        X, y = X.cuda(), y.cuda().long()
        if y.dim() == 2:
            y = y.squeeze(1)
        robust_ce, robust_err = robust_loss(
            model,
            epsilon,
            Variable(X, volatile=True),
            Variable(y, volatile=True),
            alpha_grad=True,
            scatter_grad=True,
        )

        if robust_err == 0:
            robust.append(i)
        else:
            non_robust.append(i)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum() / X.size(0)

        print(
            epoch, i, robust_ce.data.item(), robust_err, ce.data.item(), err, file=log
        )
        if i % verbose == 0:
            print(epoch, i, robust_ce.data.item(), robust_err, ce.data.item(), err)
        log.flush()

        del X, y, robust_ce, out, ce

    return robust, non_robust


def train_baseline(loader, model, opt, epoch, log, verbose):
    model.train()
    if epoch == 0:
        blank_state = opt.state_dict()

    for i, (X, y) in enumerate(loader):
        X, y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum() / X.size(0)

        opt.zero_grad()
        ce.backward()
        opt.step()

        print(epoch, i, ce.data[0], err, file=log)
        if i % verbose == 0:
            print(epoch, i, ce.data[0], err)
        log.flush()


def evaluate_baseline(loader, model, epoch, log, verbose):
    model.eval()
    for i, (X, y) in enumerate(loader):
        X, y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum() / X.size(0)

        print(epoch, i, ce.data[0], err, file=log)
        if i % verbose == 0:
            print(epoch, i, ce.data[0], err)
        log.flush()
