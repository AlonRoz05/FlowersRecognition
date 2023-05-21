import torch
import torchvision
import torch.nn as nn

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transform = weights.transforms()

model_name = "Flower_Detection_Model"
class_names = ['Daisy', 'Iris', 'Lavender', 'Lily', 'Marigold', 'Orchid', 'Poppy', 'Rose', 'Sunflower']

loaded_model = torchvision.models.efficientnet_b0(weights=weights)

for param in loaded_model.features.parameters():
    param.requires_grad = False

loaded_model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=len(class_names)))

loaded_model.load_state_dict(torch.load(f"./models/{model_name}.pth"))

def get_image_data(image_path: str):
    transformed_image = auto_transform(torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).type(torch.float32) / 255.).unsqueeze(0)

    loaded_model.eval()
    with torch.inference_mode():
        model_choice = loaded_model(transformed_image)
        choice_probs = torch.softmax(model_choice, dim=1)

        if torch.max(choice_probs) < 0.5:
            return "Unknown", (1 - torch.max(choice_probs).cpu().item()) * 100
        else:
            return torch.argmax(choice_probs, dim=1).cpu().item(), torch.max(choice_probs).cpu().item() * 100