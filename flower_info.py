import torch
import torchvision
import torch.nn as nn
import wikipediaapi

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transform = weights.transforms()

model_name = "Flower_Detection_Model"
class_names = ['Daisy', 'Iris', 'Lavender', 'Lily', 'Marigold', 'Orchid', 'Poppy', 'Rose', 'Sunflower']
flowers_families = ["Asteraceae", "Iridaceae", "Lamiaceae", "Liliaceae", "Asteraceae", "Orchidaceae", "Papaveraceae", "Rosaceae", "Asteraceae"]
non_wanted_chars = ["(", ")", ",", ".", "'", "/", ":", ";", "!", "?", " ", "!"]

wiki_api = wikipediaapi.Wikipedia('en')

loaded_model = torchvision.models.efficientnet_b0(weights=weights)

for param in loaded_model.features.parameters():
    param.requires_grad = False

loaded_model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=len(class_names)))

loaded_model.load_state_dict(torch.load(f"./models/{model_name}.pth"))

def get_flower_info(flower_img):
    transformed_image = auto_transform(torchvision.transforms.PILToTensor()(flower_img).type(torch.float32) / 255.).unsqueeze(0)

    loaded_model.eval()
    with torch.inference_mode():
        model_choice = loaded_model(transformed_image)
        choice_probs = torch.softmax(model_choice, dim=1)

    if torch.max(choice_probs) < 0.5:
        return "Unknown", "Unknown", "Unknown", "No Wikipedia page available for this flower.", f"{((1 - torch.max(choice_probs).cpu().item()) * 100):.2f}%"

    models_choice = torch.argmax(choice_probs, dim=1).cpu().item()
    model_confidence = torch.max(choice_probs).cpu().item() * 100

    if model_confidence == 100:
        model_confidence = 99.99

    flower_name = class_names[models_choice]
    page = wiki_api.page(flower_name)

    flower_family = flowers_families[models_choice]
    flower_information = page.summary
    flower_page_url = page.fullurl

    if flower_information[250:251]:
        for non_wanted_char in non_wanted_chars:
            if non_wanted_char == flower_information[249:250]:
                flower_information = flower_information[0:249] + " " + flower_information[250:]
        flower_information = f"{flower_information[0:250]}..."

    if flower_information.find("()") != -1:
        flower_information = flower_information.replace("()", "")

    return flower_name, flower_family, flower_information, flower_page_url, f"{model_confidence:.2f}%"
