# CIFAR-10
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import Generator

from utils import set_seed


def get_dataloader(batch_size=128, seed=42):
    # Load CIFAR-10 with fixed seed for reproducibility
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)

    generator = Generator().manual_seed(seed)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=2, worker_init_fn=set_seed,
                              generator=generator)
    test_loader = DataLoader(testset, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return train_loader, test_loader
