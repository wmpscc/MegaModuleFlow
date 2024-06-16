import torch
import numpy as np
import random
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, args, batch_size=32, epochs=10, seed=None):
        self.args = args
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.model_provider = None
        self.optimizer = None
        self.lr_scheduler = None

    @abstractmethod
    def setup(self, *extra_args, **kwargs):
        # set distributed environment
        pass

    @abstractmethod
    def train(self, *extra_args, **kwargs):
        # get data
        # train()
        # zero_grad()
        # forward()
        # backward()
        pass

    @abstractmethod
    def build_train_valid_test_data_iterators(self):
        # provide training set, val set and test set dataloader
        pass

    @abstractmethod
    def forward_propagation(self, batch, model):
        # return loss
        pass

    @abstractmethod
    def configure_model_and_optimizers(self):
        """
            Initialize the model, optimizers, and learning rate scheduler.

            This method sets up the following components:

            1. Model: The neural network model that will be trained. This can be a pre-trained model loaded from a checkpoint, or a newly initialized model.
            2. Optimizer(s): The optimization algorithm(s) used to update the model parameters during training, such as SGD, Adam, etc.
            3. Learning rate scheduler: A scheduler that dynamically adjusts the learning rate during training to improve convergence.
            4. (Optional) Training state: If available, the method can also load the training state from a previous checkpoint, including the model weights, optimizer state, and scheduler state.

            The specific configurations, such as the model architecture, optimizer hyperparameters, and learning rate scheduler settings, should be defined in the subclass implementation of this method.
            return self.model, self.optimizer, self.lr_scheduler
        """
        # provide model, optimizer and lr scheduler
        pass

    @staticmethod
    def load_checkpoint(model, ckpt_path=None):
        if hasattr(model, "module"):
            msg = model.module.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        else:
            msg = model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        print('Loading pretrained weight:', ckpt_path, msg)
        return model

    def set_seed(self, seed=None):
        if seed is None:
            seed = self.seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def start(self):
        self.set_seed()  # set rng
        self.setup()  # set distributed environment
        self.train()  # training
