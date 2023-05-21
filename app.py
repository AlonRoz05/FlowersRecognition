import os
import flower_info
from PIL import Image
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from flask import Flask, render_template, send_from_directory, request

app = Flask(__name__)
app.config["SECRET_KEY"] = "}iAFozy.C81Rqa7jA]t,Bh+U6JHaq3mV5k=E1fTCt@Xy6]kMi~*71::MpEB>.#zwVv.RnuaW^,Td}n1Q*4X_n:]X-a>^NHbo?1Kx"
app.config["UPLOAD_FOLDER"] = "static/uploaded_imgs"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]

class UploadImageForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Submit")

@app.route('/', methods=['GET', 'POST'])
def home():
    clear_image()

    form = UploadImageForm()

    if request.method == "POST":
        if request.form.get("clear-image"):
            clear_image()
            return render_template("index.html", form=form, flower_name="", flower_family="", flower_information="", models_choice_probs="")
        

    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config["UPLOAD_FOLDER"], secure_filename("image.png")))
        
        flower_name, flower_family, flower_information, flower_page_url, model_confidence = flower_info.get_flower_info(flower_img_path="static/uploaded_imgs/image.png")
        
        if flower_name == "Unknown":
            return render_template("index.html", form=form, flower_name=flower_name, flower_family=flower_family, flower_information=flower_information, models_choice_probs=model_confidence)

        return render_template("index.html", form=form, flower_name=flower_name, flower_family=flower_family, flower_information=flower_information, models_choice_probs=model_confidence)

    return render_template("index.html", form=form, flower_name="", flower_family="", flower_information="", models_choice_probs="")

@app.route('/serve-img', methods=['GET'])
def serve_img():
    return send_from_directory(app.config["UPLOAD_FOLDER"], "image.png")

def clear_image():
    background_image = Image.open("static/important_imgs/background.png")
    background_image.save("static/uploaded_imgs/image.png")

if __name__ == "__main__":
    app.run(debug=True, port=8000)
