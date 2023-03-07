import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timeit import default_timer as timer
from pathlib import Path
from PIL import Image
import helper_FRFN as helper
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

img_path = Path("Data")
train_dir = img_path / "Training_Data"
test_dir = img_path / "Testing_Data"

img_path_list = list(img_path.glob("*/*/*.jpg"))
random_img_path = random.choice(img_path_list)
img_class = random_img_path.parent.stem

img = Image.open(random_img_path)

data_transform_train = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

data_transform_test = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform_train)
test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transform_test)

BATCH_SIZE = 32
class_names = train_dataset.classes

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class Flowers_NN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13, out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

model = Flowers_NN(input_shape=3, hidden_units=10, output_shape=len(class_names))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

epochs = 5
train = True
show_model_quality = True
if train:
    start_training_time = timer()

    model_results = helper.train_model(model=model, 
                                        train_dataloader=train_dataloader,
                                        test_dataloader=test_dataloader,
                                        loss_fn=loss_fn, 
                                        optimizer=optimizer,
                                        epochs=epochs,
                                        device=device)

    end_training_time = timer()
    print(f"Training took {end_training_time-start_training_time:.3f} on {device}")
    helper.plot_loss_curves(results=model_results, epochs=epochs)

    MODEL_PATH = Path("./models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "Flower_Detection_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

else:
    loaded_model = Flowers_NN(input_shape=3, hidden_units=10, output_shape=len(class_names))
    loaded_model.load_state_dict(torch.load("./models/Flower_Detection_model.pth"))
    loaded_model.to(device)

    if show_model_quality:
        test_samples = []
        test_labels = []
        for sample, label in random.sample(list(test_dataset), k=9):
            test_samples.append(sample)
            test_labels.append(label)

        pred_probs = helper.make_predictions(model=loaded_model, data=test_samples, device=device)
        pred_classes = pred_probs.argmax(dim=1)

        helper.plot_model_predictions(pred_classes=pred_classes, test_labels=test_labels, test_samples=test_samples, class_names=class_names)
        helper.p_confusion_matrix(model=loaded_model, test_data_loader=test_dataloader,data=test_dataset, class_names=class_names, device=device)

    else:
        print("in progress...")

