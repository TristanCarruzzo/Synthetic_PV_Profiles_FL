import torch

from copy import deepcopy
import sys


class Trainer:
    """
    Responsible of training and evaluating a (deep-)learning model

    Attributes
    ----------
    model (nn.Module): the model trained by the learner

    criterion (torch.nn.modules.loss): loss function used to train the `model`, should have reduction="none"

    metric (fn): function to compute the metric, should accept as input two vectors and return a scalar

    device (str or torch.device):

    optimizer (torch.optim.Optimizer):

    lr_scheduler (torch.optim.lr_scheduler):

    cast_label (bool): whether to cast labels to float or not, if `BCELoss`
    is used as criterion this should be set to True


    Methods
    -------

    optimizer_step: perform one optimizer step, requires the gradients to be already computed.

    fit_batch: perform an optimizer step over one batch

    fit_epoch:

    fit_batches: perform successive optimizer steps over successive batches

    fit_epochs:

    evaluate_iterator: evaluate `model` on an iterator

    gather_losses:

    get_param_tensor: get `model` parameters as a unique flattened tensor

    free_gradients:

    free_memory:

    """
    def __init__(
            self,
            model,
            criterion,
            metric,
            device,
            optimizer,
            lr_scheduler=None,
            cast_label=False,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.cast_label = cast_label

        self.is_ready = True

        self.n_modules = self.__get_num_modules()
        self.model_dim = int(self.get_param_tensor().shape[0])

    def __get_num_modules(self):
        """
        computes the number of modules in the model network;
        i.e., the size of `self.model.modules()`

        return:
            n_modules (int)
        """
        if not self.is_ready:
            return

        n_modules = 0
        for _ in self.model.modules():
            n_modules += 1

        return n_modules

    def fit_batch(self, batch, frozen_modules=None):
        """
        perform an optimizer step over one batch drawn from `iterator`

        :param batch: tuple of (x, y, weights)
        :param frozen_modules: list of frozen modules; default is None

        :return:
            (loss.item(), batch_weight)
            (metric.item(), batch_size)

        """
        if frozen_modules is None:
            frozen_modules = []

        self.model.train()

        x, y, weights = batch

        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)
        weights = weights.to(self.device).type(torch.float32)

        if self.cast_label:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)

        loss_vec = self.criterion(y_pred, y)
        metric = self.metric(y_pred, y)

        loss = loss_vec @ weights

        for frozen_module in frozen_modules:
            frozen_module.zero_grad()

        loss.backward()

        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        batch_weight = weights.sum().item() + sys.float_info.epsilon

        return (loss.item() / batch_weight, batch_weight), (metric.item(), len(weights))

    def fit_epoch(self, iterator, frozen_modules=None):
        """
        perform several optimizer steps on all batches drawn from `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param frozen_modules: list of frozen models; default is None

        :return:
            (loss.item(), weights_sum)
            (metric.item(), n_samples)

        """
        if frozen_modules is None:
            frozen_modules = []

        self.model.train()

        global_loss = 0.
        weights_sum = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, weights in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)
            weights = weights.to(self.device).type(torch.float32)

            if self.cast_label:
                y = y.type(torch.float32).unsqueeze(1)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss_vec = self.criterion(y_pred, y)
            loss = loss_vec @ weights

            loss.backward()

            for frozen_module in frozen_modules:
                frozen_module.zero_grad()

            self.optimizer.step()

            global_loss += loss.item()
            weights_sum += weights.sum().item() + sys.float_info.epsilon
            global_metric += self.metric(y_pred, y).item() * y.size(0)
            n_samples += y.size(0)

        return (global_loss / weights_sum, weights_sum), (global_metric / n_samples, n_samples)

    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader

        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for x, y, _ in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                if self.cast_label:
                    y = y.type(torch.float32).unsqueeze(1)

                y_pred = self.model(x)

                global_loss += self.criterion(y_pred, y).mean().item() * y.size(0)
                global_metric += self.metric(y_pred, y).item() * y.size(0)

                n_samples += y.size(0)

        return global_loss / n_samples, global_metric / n_samples, n_samples

    def fit_batches(self, iterator, n_steps, frozen_modules=None):
        """
        perform successive optimizer steps over successive batches drawn from iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_steps: number of successive batches
        :type n_steps: int
        :param frozen_modules: list of frozen models; default is None

        :return:
            (loss.item(), global_weights_sum)
            (metric.item(), n_samples)

        """
        global_loss = 0.
        global_weights_sum = 0.
        global_metric = 0.
        total_n_samples = 0

        for step in range(n_steps):
            (batch_loss, batch_weight), (batch_metric, n_samples) = \
                self.fit_batch(iterator, frozen_modules=frozen_modules)

            global_loss += batch_loss * batch_weight
            global_weights_sum += batch_weight
            global_metric += batch_metric * n_samples
            total_n_samples += n_samples

        return \
            (global_loss / global_weights_sum, global_weights_sum), (global_metric / total_n_samples, total_n_samples)

    def fit_epochs(self, iterator, n_epochs, frozen_modules=None):
        """
        perform multiple training epochs

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of successive batches
        :type n_epochs: int
        :param frozen_modules: list of frozen models; default is None

        :return:
            (loss.item(), weights_sum)
            (metric.item(), n_samples)

        """
        global_loss = 0.
        global_weights_sum = 0.
        global_metric = 0.
        total_n_samples = 0

        for step in range(n_epochs):
            (epoch_loss, epoch_weights_sum), (epoch_metric, n_samples) = \
                self.fit_epoch(iterator, frozen_modules=frozen_modules)

            global_loss += epoch_loss * epoch_weights_sum
            global_weights_sum += epoch_weights_sum
            global_metric += epoch_metric * n_samples
            total_n_samples += n_samples

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return \
            (global_loss / global_weights_sum, global_weights_sum), (global_metric / total_n_samples, total_n_samples)

    def get_param_tensor(self):
        """
        get `model` parameters as a unique flattened tensor

        :return: torch.tensor

        """
        param_list = []

        for param in self.model.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)

    def set_param_tensor(self, param_tensor):
        """
        sets the parameters of the model from `param_tensor`

        :param param_tensor: torch.tensor of shape (`self.model_dim`,)

        """
        param_tensor = param_tensor.to(self.device)

        current_index = 0
        for param in self.model.parameters():
            param_shape = param.data.shape
            current_dimension = param.data.view(-1, ).shape[0]

            param.data = \
                param_tensor[current_index: current_index + current_dimension].reshape(param_shape)

            current_index += current_dimension

    def get_grad_tensor(self):
        """
        get `model` gradients as a unique flattened tensor

        :return: torch.tensor

        """
        grad_list = []

        for param in self.model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)

    def set_grad_tensor(self, grad_tensor):
        """

        :param grad_tensor: torch.tensor of shape (`self.model_dim`,)

        """
        grad_tensor = grad_tensor.to(self.device)

        current_index = 0
        for param in self.model.parameters():
            param_shape = param.data.shape
            current_dimension = param.data.view(-1, ).shape[0]

            param.grad.data = \
                deepcopy(grad_tensor[current_index: current_index + current_dimension].reshape(param_shape))

            current_index += current_dimension

    def free_gradients(self):
        """
        free memory allocated by gradients

        """

        self.optimizer.zero_grad(set_to_none=True)

    def free_memory(self):
        """
        free the memory allocated by the model weights

        """
        if not self.is_ready:
            return

        self.free_gradients()

        del self.lr_scheduler
        del self.optimizer
        del self.model

        self.is_ready = False
