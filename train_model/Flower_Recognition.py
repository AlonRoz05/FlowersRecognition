import torch
import torchvision
import torch.nn as nn
import data_setup
import train_save
import helper_FRFN

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else device

train_dir = "Data/Main_Flower_Dataset/Training_Data"
test_dir = "Data/Main_Flower_Dataset/Testing_Data"

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transform = weights.transforms()

train_dataloader, test_dataloader, test_dataset_no_transform, class_names = data_setup.create_dataloaders(train_dir=train_dir, 
                                                                                             test_dir=test_dir, 
                                                                                             transform=auto_transform,
                                                                                             batch_size=64)
epochs = 5
input_shape = 3
hidden_units = 10
output_shape = len(class_names)
model_name = "Flower_Detection_Model"

model = torchvision.models.efficientnet_b0(weights=weights).to(device)

for param in model.features.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=output_shape)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

train = False
show_model_results = False
if train:
    print(f"\n[INFO] Training model on {device.upper()}.")
    train_save.train_save_model(model=model, 
                                model_save_name=model_name, 
                                train_dataloader=train_dataloader, 
                                test_dataloader=test_dataloader, 
                                loss_fn=loss_fn, 
                                optimizer=optimizer, 
                                epochs=epochs, 
                                device=device,
                                writer=writer)

if show_model_results:
    loaded_model = torchvision.models.efficientnet_b0(weights=weights)
    
    loaded_model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=output_shape)).to(device)

    loaded_model.load_state_dict(torch.load(f"./models/{model_name}.pth"))
    loaded_model.to(device)

    helper_FRFN.show_model_quality(model=loaded_model, test_dataloader=test_dataloader, test_dataset=test_dataset_no_transform, class_names=class_names, device=device)
