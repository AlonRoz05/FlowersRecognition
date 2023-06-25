import os
import flower_info
from PIL import Image
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from flask import Flask, render_template, send_from_directory, redirect

app = Flask(__name__)
app.config["SECRET_KEY"] = "4A0h-bayPsar:tF>,8ffgzGDQPJLya+FR2NtwXbtC^kA]vJp-b3a40cJoeC36~Y)Rg8+duz>4U2*ZN^=-r5f))Wxf*b3n10V>U4y"
app.config["UPLOAD_FOLDER"] = "static/uploaded_imgs"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]

uploaded_img_name = "image.png"

background_image = Image.open("static/background.png")
if os.path.isfile("static/uploaded_imgs/flower_from_gallery_img.png"):
    os.remove("static/uploaded_imgs/flower_from_gallery_img.png")
background_image.save("static/uploaded_imgs/image.png")

class UploadImageForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Submit")

@app.route('/', methods=['GET', 'POST'])
def home():
    global flower_page_url, uploaded_img_name

    form = UploadImageForm()

    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config["UPLOAD_FOLDER"], secure_filename("image.png")))
        uploaded_img_name = "image.png"

        flower_name, flower_family, flower_information, flower_page_url, model_confidence = flower_info.get_flower_info(flower_img="static/uploaded_imgs/image.png")

        return render_template("index.html", form=form, flower_name=flower_name, flower_family=flower_family, flower_information=flower_information, model_confidence=model_confidence, flower_page_url=flower_page_url)

    if os.path.isfile("static/uploaded_imgs/flower_from_gallery_img.png"):
        flower_name, flower_family, flower_information, flower_page_url, model_confidence = flower_info.get_flower_info(flower_img="static/uploaded_imgs/flower_from_gallery_img.png")
        return render_template("index.html", form=form, flower_name=flower_name, flower_family=flower_family, flower_information=flower_information, model_confidence=model_confidence, flower_page_url=flower_page_url)

    return render_template("index.html", form=form, flower_name="", flower_family="", flower_information="", model_confidence="", flower_page_url="")

@app.route('/serve-img', methods=['GET'])
def serve_img():
    return send_from_directory(app.config["UPLOAD_FOLDER"], uploaded_img_name)

@app.route('/go-to-flower-page')
def put_flower_page():
    try:
        return redirect(flower_page_url)
    except:
        return redirect("http://127.0.0.1:8000")

@app.route('/flower_1')
def flower_1():
    global uploaded_img_name
    flower_img = Image.open("static/flowers_imgs/lavender_img.png")
    flower_img.save("static/uploaded_imgs/flower_from_gallery_img.png")
    uploaded_img_name = "flower_from_gallery_img.png"
    return redirect("http://127.0.0.1:8000")

@app.route('/flower_2')
def flower_2():
    global uploaded_img_name
    flower_img = Image.open("static/flowers_imgs/rose_img.png")
    flower_img.save("static/uploaded_imgs/flower_from_gallery_img.png")
    uploaded_img_name = "flower_from_gallery_img.png"
    return redirect("http://127.0.0.1:8000")

@app.route('/flower_3')
def flower_3():
    global uploaded_img_name
    flower_img = Image.open("static/flowers_imgs/iris_img.png")
    flower_img.save("static/uploaded_imgs/flower_from_gallery_img.png")
    uploaded_img_name = "flower_from_gallery_img.png"
    return redirect("http://127.0.0.1:8000")

@app.route('/flower_4')
def flower_4():
    global uploaded_img_name
    flower_img = Image.open("static/flowers_imgs/poppy_img.png")
    flower_img.save("static/uploaded_imgs/flower_from_gallery_img.png")
    uploaded_img_name = "flower_from_gallery_img.png"
    return redirect("http://127.0.0.1:8000")

@app.route('/flower_5')
def flower_5():
    global uploaded_img_name
    flower_img = Image.open("static/flowers_imgs/sunflower_img.png")
    flower_img.save("static/uploaded_imgs/flower_from_gallery_img.png")
    uploaded_img_name = "flower_from_gallery_img.png"
    return redirect("http://127.0.0.1:8000")

if __name__ == "__main__":
    app.run(debug=True, port=8000)