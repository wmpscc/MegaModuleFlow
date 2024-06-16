import os
import deepspeed
import torch
from torch.utils.data import DataLoader
from trainer.trainer_base import BaseTrainer


class DeepSpeedTrainer(BaseTrainer):
    def __init__(self, args, model_provider, train_valid_test_datasets_provider, batch_size=32, epochs=10, seed=0):
        super().__init__(args, batch_size, epochs, seed)
        self.model_provider = model_provider
        self.datasets_provider = train_valid_test_datasets_provider
        self.world_size = args.world_size
        self.args = args

    def setup(self):
        deepspeed.init_distributed()

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
        model = self.model_provider()
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=self.args.deepspeed_config
        )
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
                model.backward(loss)
                optimizer.step()
            lr_scheduler.step()

    def start(self):
        self.setup()
        self.train()


if __name__ == "__main__":
    # Define your model provider and dataset provider
    def model_provider():
        return MyModel()


    def dataset_provider():
        return MyTrainDataset(), MyValidDataset(), MyTestDataset()


    # Arguments for the trainer
    class Args:
        world_size = 4
        deepspeed_config = "config/deepspeed_config.json"


    args = Args()

    # Instantiate and start the DeepSpeedTrainer
    trainer = DeepSpeedTrainer(args, model_provider, dataset_provider, batch_size=32, epochs=10, seed=42)
    trainer.start()
