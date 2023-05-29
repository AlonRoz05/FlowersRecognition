import flower_info
from PIL import Image
from io import BytesIO
from fastapi import FastAPI
from base64 import b64decode

app = FastAPI()

def decode_imgs(cFile: str):
    dFile = cFile.replace("_", "/").encode()
    img = Image.open(BytesIO(b64decode(dFile + b"==")))
    return img

@app.get("/get_flower_info/{language}/{b64File}")
async def return_flower_info(language: str, b64File: str):
    if language == "english":
        flower_name, flower_family, flower_information, flower_page_url, model_confidence = flower_info.get_flower_info(decode_imgs(b64File))
        return {"flower_name": flower_name, "flower_family": flower_family, "flower_information": flower_information, "flower_page_url": flower_page_url, "model_confidence": model_confidence}
    else:
        return {"flower_name": "Unsupported language"}
