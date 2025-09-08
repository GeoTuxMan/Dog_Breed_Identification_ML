import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)

# Încărcăm modelul
model = tf.keras.models.load_model("dog_breed_final.h5")

# Încărcăm mapping-ul claselor
class_indices = {}
with open("class_indices.txt", "r") as f:
    for line in f:
        idx, breed = line.strip().split(":")
        class_indices[int(idx)] = breed

class DogBreedApp:
    def __init__(self, master):
        self.master = master
        master.title("Dog Breed Classifier")

        self.label = tk.Label(master, text="Upload a dog image to classify its breed")
        self.label.pack()

        self.upload_btn = tk.Button(master, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()

        self.canvas = tk.Canvas(master, width=400, height=300)
        self.canvas.pack()

        self.classify_btn = tk.Button(master, text="Classify Breed", command=self.classify_image)
        self.classify_btn.pack()

        self.image_path = None
        self.tk_image = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            self.image_path = file_path
            pil_image = Image.open(file_path).resize((400, 300))
            self.tk_image = ImageTk.PhotoImage(pil_image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def classify_image(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return

        # Preprocesăm imaginea
        img = Image.open(self.image_path).resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prezicem
        preds = model.predict(img_array)
        class_idx = np.argmax(preds)
        breed = class_indices[class_idx]

        messagebox.showinfo("Prediction", f"This dog looks like a: {breed}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DogBreedApp(root)
    root.mainloop()
