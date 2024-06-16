import torch
from torch.utils.data import DataLoader
from trainer.trainer_base import BaseTrainer


class SingleTrainer(BaseTrainer):
    def __init__(self, args, model_provider, train_valid_test_datasets_provider, batch_size=32, epochs=10, seed=0):
        super().__init__(args, batch_size, epochs, seed)
        self.model_provider = model_provider
        self.datasets_provider = train_valid_test_datasets_provider

    def setup(self):
        pass

    def build_train_valid_test_data_iterators(self):
        train_dataset, valid_dataset, test_dataset = self.datasets_provider()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        valid_loader = None
        if valid_dataset is not None:
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader

    def configure_model_and_optimizers(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model_provider()
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return model, optimizer, lr_scheduler

    def forward_propagation(self, batch, model):
        inputs, labels = batch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        return loss

    def train(self):
        self.set_seed()
        train_loader, valid_loader, test_loader = self.build_train_valid_test_data_iterators()
        model, optimizer, lr_scheduler = self.configure_model_and_optimizers()

        for epoch in range(self.epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                loss = self.forward_propagation(batch, model)
                loss.backward()
                optimizer.step()
            lr_scheduler.step()

    def start(self):
        self.setup()
        self.train()
