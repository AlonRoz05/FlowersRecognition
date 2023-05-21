import wikipediaapi
import model_builder

wiki = wikipediaapi.Wikipedia('en')
class_names = model_builder.class_names

def get_flower_info(flower_img_path: str):
    models_choice, model_confidence = model_builder.get_image_data(image_path=flower_img_path)

    if model_confidence == 100:
        model_confidence = 99.99

    flower_name = class_names[models_choice]
    if models_choice == "Unknown":
        flower_name = "Unknown"
        return "Unknown", "Unknown", "Unknown", "No Wikipedia page available for this flower.", f"{model_confidence:.2f}%"

    page = wiki.page(flower_name)

    flower_family = ""
    flower_information = page.summary
    flower_page_url = page.fullurl

    if flower_information[250:251]:
        flower_information = f"{flower_information[0:250]}..."

    if flower_information.find("()") != -1:
        flower_information = flower_information.replace("()", "")

    return flower_name, flower_family, flower_information, flower_page_url, f"{model_confidence:.2f}%"
