import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import json

# Lataa malli
MODEL_PATH = "dog_breed_inception.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Lataa idx_to_breed JSON-tiedostosta
IDX_TO_BREED_PATH = "idx_to_breed.json"
with open(IDX_TO_BREED_PATH, "r") as f:
    idx_to_breed = json.load(f)

# Kuvan esikäsittely
IMG_SIZE = (299, 299)  # InceptionV3:n suosituskoko

def preprocess_image(img_path):
    """
    Lataa kuvan, skaalaa oikeaan kokoon ja normalisoi.
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0  # Skaalaus [0, 1]
    return np.expand_dims(img_array, axis=0)

# Ennustefunktio
def predict_breed(img_path):
    """
    Ennustaa koiran rodun ja palauttaa sen nimen.
    """
    input_arr = preprocess_image(img_path)
    preds = model.predict(input_arr)
    class_idx = np.argmax(preds, axis=1)[0]
    return idx_to_breed.get(str(class_idx), "Unknown breed")

# Tkinter GUI -käyttöliittymä
def open_image():
    """
    Valitsee kuvan, näyttää sen ja näyttää ennustetun koirarodun.
    """
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if not file_path:
        return  # Käyttäjä ei valinnut tiedostoa

    # Ennustetaan koirarotu
    breed = predict_breed(file_path)

    # Näytetään kuva käyttöliittymässä
    img = Image.open(file_path).resize((200, 200))  # Esikatselukuvan koko
    tk_img = ImageTk.PhotoImage(img)
    image_label.config(image=tk_img)
    image_label.image = tk_img  # Pidä viite

    # Näytetään rotutulos
    result_label.config(text=f"Koiran rotu: {breed}")

# Tkinter-ikkunan asetukset
root = tk.Tk()
root.title("Dog Breed Classifier")

# Painike kuvan valitsemiseen
open_button = tk.Button(root, text="Valitse kuva", command=open_image, font=("Arial", 12))
open_button.pack(pady=10)

# Alue kuvan näyttämiselle
image_label = tk.Label(root)
image_label.pack()

# Alue tuloksen näyttämiselle
result_label = tk.Label(root, text="Ei kuvaa ladattu.", font=("Arial", 14))
result_label.pack(pady=10)

# Käynnistä Tkinter-silmukka
root.mainloop()
