from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_dataloaders(train_dir, test_dir, transform: transforms.Compose, batch_size: int):
    train_dataset = datasets.ImageFolder(root=train_dir,
                                        transform=transform)

    test_dataset = datasets.ImageFolder(root=test_dir,
                                        transform=transform)

    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    class_names = train_dataset.classes

    return train_dataloader, test_dataloader, test_dataset, class_names
