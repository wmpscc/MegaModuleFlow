import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from trainer.trainer_base import BaseTrainer
from torch.utils.data import DataLoader


class DDPTrainer(BaseTrainer):
    def __init__(self, args, model_provider, train_valid_test_datasets_provider, batch_size=32, epochs=10, seed=0):
        super().__init__(args, batch_size, epochs, seed)
        self.model_provider = model_provider
        self.datasets_provider = train_valid_test_datasets_provider
        self.world_size = args.world_size

    def setup(self, rank):
        "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
        os.environ["MASTER_ADDR"] = 'localhost' if self.args.master_addr is None else self.args.master_addr
        os.environ["MASTER_PORT"] = "12355" if self.args.master_port is None else self.args.master_port
        dist.init_process_group("nccl", rank=rank, world_size=self.world_size)
        self.global_rank = rank

    def build_train_valid_test_data_iterators(self):
        train_dataset, valid_dataset, test_dataset = self.datasets_provider()
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.global_rank)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler)

        if valid_dataset is not None:
            valid_sampler = DistributedSampler(valid_dataset, num_replicas=self.world_size, rank=self.global_rank)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, sampler=valid_sampler)
        else:
            valid_loader = None
        if test_dataset is not None:
            test_sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=self.global_rank)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=test_sampler)
        else:
            test_loader = None
        return train_loader, valid_loader, test_loader

    def configure_model_and_optimizers(self):
        # need implement
        model = self.model_provider()
        model = model.to(self.global_rank)
        model = DDP(model, device_ids=[self.global_rank])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return model, optimizer, lr_scheduler

    def forward_propagation(self, batch, model):
        # need implement
        inputs, labels = batch
        inputs = inputs.to(self.global_rank)
        labels = labels.to(self.global_rank)
        loss = model(inputs, labels)
        return loss

    def train(self, rank, args):
        self.setup(rank)
        self.set_seed()
        train_loader, valid_loader, test_loader = self.build_train_valid_test_data_iterators()
        model, optimizer, lr_scheduler = self.configure_model_and_optimizers()

        for epoch in range(self.epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                loss = self.forward_propagation(batch, model)
                loss.backward()
                optimizer.step()
            lr_scheduler.step()

        dist.destroy_process_group()

    def start(self):
        mp.spawn(self.train, args=(self.args,), nprocs=self.world_size, join=True)


if __name__ == "__main__":
    # Define your model provider and dataset provider
    def model_provider():
        return MyModel()


    def dataset_provider():
        return MyTrainDataset(), MyValidDataset(), MyTestDataset()


    # Arguments for the trainer
    class Args:
        world_size = 4
        master_addr = 'localhost'
        master_port = '12355'


    args = Args()

    # Instantiate and start the DDPTrainer
    trainer = DDPTrainer(args, model_provider, dataset_provider, batch_size=32, epochs=10, seed=42)
    trainer.start()
