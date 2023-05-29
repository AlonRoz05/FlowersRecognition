const dropdowns = document.querySelectorAll('.Language-selector');
let active = document.querySelector('.active');
let active_language_text = active.innerText;

const inpFile = document.getElementById("inpFile");

let API_url = "http://127.0.0.1:8000/get_flower_info/" + active_language_text.toLowerCase() + "/";

let base64_image_url = "";

inpFile.addEventListener("change", e => {
    const file = inpFile.files[0];
    const reader = new FileReader();

    reader.addEventListener("load", () => {
        base64_image_url = reader.result;

        document.getElementById("uploaded_image").setAttribute('src', base64_image_url);
        base64_image_url = base64_image_url.slice(22);

        while (base64_image_url.includes("/")){
            base64_image_url = base64_image_url.replace("/", "_");
        }

        API_url = API_url + base64_image_url;
        getFlowerInfo();
    });

    reader.readAsDataURL(file);
});

async function getFlowerInfo() {
    const response = await fetch(API_url);
    const data = await response.json();
    const { flower_name } = data;
    if (flower_name != "Unsupported language") {
        const { flower_name, flower_family, flower_information, model_confidence, flower_page_url } = data;
        document.getElementById("flower_name").textContent = flower_name;
        document.getElementById("flower_family").textContent = flower_family;
        document.getElementById("flower_information").textContent = flower_information;
        document.getElementById("model_confidence").textContent = model_confidence;
        document.getElementById("flower_page_link").setAttribute('href', flower_page_url);
    }
    else {
        console.log("Unsupported language");
    }
}

dropdowns.forEach(dropdown => {
    const select = dropdown.querySelector('.select');
    const caret = dropdown.querySelector('.caret');
    const menu = dropdown.querySelector('.menu');
    const options = dropdown.querySelectorAll('.menu li');
    const selected = dropdown.querySelector('.selected');

    select.addEventListener('click', () => {
        select.classList.toggle('select-clicked');
        caret.classList.toggle('caret-rotate');
        menu.classList.toggle('menu-open');

    });

    options.forEach(option => {
        option.addEventListener('click', () => {
            selected.innerText = option.innerText;
            select.classList.remove('select-clicked');
            caret.classList.remove('caret-rotate');
            menu.classList.remove('menu-open');
            options.forEach(option => {
                option.classList.remove('active');
            });
            option.classList.add('active');
            active = document.querySelector('.active');
        });
    });
});
