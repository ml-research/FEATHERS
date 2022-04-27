import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
from copy import deepcopy
#from helpers import Logger, compute_accuracy
import warnings
from tensorboardX import SummaryWriter
from datetime import datetime as dt

warnings.filterwarnings('error')

class DartsTrainer:
    def __init__(self, model, loss, dataset_train, dataset_valid,
                 batch_size=64, device=None, arc_learning_rate=3.0E-4, second_order_optim=False):

        self.device = device if device is not None else torch.device("cpu")
        self.model = model
        self.loss = loss
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        date = dt.strftime(dt.now(), '%Y:%m:%d:%H:%M:%S')
        self.writer = SummaryWriter("./runs/Client_{}".format(date))
        
        self.ctrl_optim = torch.optim.Adam(self.model.arch_parameters(), arc_learning_rate, betas=(0.5, 0.999),
                                           weight_decay=1.0E-5)
        #self.ctrl_optim = torch.optim.SGD(self.model.arch_parameters(), arc_learning_rate, 0.9)
        self.second_order_optim = second_order_optim

        self.train_loader = DataLoader(self.dataset_train,
                                        batch_size=batch_size,
                                        pin_memory=True,
                                        num_workers=2)
        self.valid_loader = DataLoader(self.dataset_valid,
                                        batch_size=batch_size,
                                        pin_memory=True,
                                        num_workers=2)
        self.optimizer = None

    def set_current_hyperparameter_config(self, hyperparam, idx):
        self.hyperparam_config = hyperparam
        self.hyperparam_idx = idx
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.hyperparam_config['learning_rate'], 0.9, weight_decay=0)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = self.hyperparam_config['learning_rate']

    
    def train_one_epoch(self, epoch):
        # self.optimizer = torch.optim.SGD(self.model.parameters(), self.hyperparam_config, 0.9, weight_decay=0)
        self.model.train()
        running_val_loss, running_train_loss = 0, 0

        # collect current loss before update
        before_loss = self.validate_one_epoch(epoch)
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.train_loader, self.valid_loader)):
            # print("Batch {}/{}".format(step + 1, len(self.train_loader)), end='\r')
            trn_X, trn_y = trn_X.to(self.device, non_blocking=True), trn_y.to(self.device, non_blocking=True)
            val_X, val_y = val_X.to(self.device, non_blocking=True), val_y.to(self.device, non_blocking=True)

            # get current validation loss and logits
            val_logits, val_loss = self._logits_and_loss(val_X, val_y)
            self.writer.add_histogram('val_logits', val_logits, epoch)

            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            if self.second_order_optim:
                self._unrolled_backward(trn_X, trn_y, val_X, val_y)
            else:
                self._backward(val_X, val_y)
            self.ctrl_optim.step()

            # phase 2: child network step
            self.optimizer.zero_grad()
            logits, loss = self._logits_and_loss(trn_X, trn_y)
            self.writer.add_histogram('train_logits', logits, epoch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 15.)  # gradient clipping
            # log gradients of last 5 layers
            for name, param in list(reversed(list(self.model.named_parameters())))[:10]:
                self.writer.add_histogram(name + '.grad', param.grad, epoch*step)
            self.optimizer.step()
            # log weights of last 5 layers
            for name, param in list(reversed(list(self.model.named_parameters())))[:10]:
                self.writer.add_histogram(name, param.data, epoch*step)

            running_train_loss += loss.item()
            running_val_loss += val_loss.item()
            
        self.writer.add_scalar("Validation Loss", running_val_loss, epoch)
        self.writer.add_scalar("Training Loss", running_train_loss, epoch)

        # collect loss after update
        after_loss = self.validate_one_epoch(epoch)

        return self.hyperparam_config, self.hyperparam_idx, before_loss.item(), after_loss.item()

    def validate_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0
        with torch.no_grad():
            for step, (X, y) in enumerate(self.valid_loader):
                X, y = X.to(self.device), y.to(self.device)
                logits, loss = self._logits_and_loss(X, y)
                running_loss += loss
        return running_loss

    def _logits_and_loss(self, X, y):
        logits = self.model(X)
        loss = self.loss(logits, y)
        return logits, loss

    def _backward(self, val_X, val_y):
        """
        Simple backward with gradient descent
        """
        _, loss = self._logits_and_loss(val_X, val_y)
        loss.backward()

    def _unrolled_backward(self, trn_X, trn_y, val_X, val_y):
        """
        Compute unrolled loss and backward its gradients
        """
        backup_params = deepcopy(tuple(self.model.parameters()))

        # do virtual step on training data
        lr = self.optimizer.param_groups[0]["lr"]
        momentum = self.optimizer.param_groups[0]["momentum"]
        weight_decay = self.optimizer.param_groups[0]["weight_decay"]
        self._compute_virtual_model(trn_X, trn_y, lr, momentum, weight_decay)

        # calculate unrolled loss on validation data
        # keep gradients for model here for compute hessian
        _, loss = self._logits_and_loss(val_X, val_y)
        w_model, w_ctrl = tuple(self.model.parameters()), tuple(self.model.arch_parameters())
        w_grads = torch.autograd.grad(loss, w_model + w_ctrl)
        d_model, d_ctrl = w_grads[:len(w_model)], w_grads[len(w_model):]

        # compute hessian and final gradients
        hessian = self._compute_hessian(backup_params, d_model, trn_X, trn_y)
        with torch.no_grad():
            for param, d, h in zip(w_ctrl, d_ctrl, hessian):
                # gradient = dalpha - lr * hessian
                param = param.to(self.device)
                d = d.to(self.device)
                param.grad = d - lr * h

        # restore weights
        self._restore_weights(backup_params)

    def _compute_virtual_model(self, X, y, lr, momentum, weight_decay):
        """
        Compute unrolled weights w`
        """
        # don't need zero_grad, using autograd to calculate gradients
        _, loss = self._logits_and_loss(X, y)
        gradients = torch.autograd.grad(loss, self.model.parameters())
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), gradients):
                m = self.optimizer.state[w].get("momentum_buffer", 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)

    def _restore_weights(self, backup_params):
        with torch.no_grad():
            for param, backup in zip(self.model.parameters(), backup_params):
                param.copy_(backup)

    def _compute_hessian(self, backup_params, dw, trn_X, trn_y):
        """
            dw = dw` { L_val(w`, alpha) }
            w+ = w + eps * dw
            w- = w - eps * dw
            hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
            eps = 0.01 / ||dw||
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        dalphas = []
        for e in [eps, -2. * eps]:
            # w+ = w + eps*dw`, w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), dw):
                    p += e * d

            _, loss = self._logits_and_loss(trn_X, trn_y)
            dalphas.append(torch.autograd.grad(loss, self.model.arch_parameters()))

        dalpha_pos, dalpha_neg = dalphas  # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
        hessian = []
        for p, n in zip(dalpha_pos, dalpha_neg):
            p, n = p.to(self.device), n.to(self.device)
            hessian.append((p - n) / 2. * eps)
        # hessian = [(p - n) / (2. * eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
