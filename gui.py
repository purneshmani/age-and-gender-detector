import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np



# Load the model
from keras.losses import MeanAbsoluteError
from keras.models import load_model

custom_objects = {"mae": MeanAbsoluteError()}

model = load_model("C:/Users/91812/Downloads/Age_Sex_Detection.h5", custom_objects=custom_objects)


# Initialize GUI
top = tk.Tk()
top.geometry('800x600') 
top.title('Age & Gender Detector')
top.configure(background='#CDCDCD')

# Labels and Image Holder
label1 = Label(top, background="#CDCDCD", font=('arial', 15, "bold"))
label2 = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
sign_image = Label(top)


def Detect(file_path):
    image = Image.open(file_path)
    image = image.resize((48, 48))  
    image = np.array(image)

    
    if image.shape[-1] != 3:
        image = np.stack((image,) * 3, axis=-1)

    image = image.astype('float32') / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make predictions
    pred = model.predict(image)

    age = int(np.round(pred[1][0]))  # Age Prediction
    sex = 1 if pred[0][0] > 0.5 else 0  # Convert probability to Male (1) / Female (0)
    
    sex_f = ["Male", "Female"]
    
    print(f"Predicted Age: {age}")
    print(f"Predicted Gender: {sex_f[sex]}")

    label1.configure(foreground="#011638", text=f"Age: {age}")
    label2.configure(foreground="#011638", text=f"Gender: {sex_f[sex]}")


def show_Detect_button(file_path):
    Detect_b = Button(top, text="Detect Image", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail((top.winfo_width()//2, top.winfo_height()//2))  
        
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.config(text='')  # Clear previous text
        label2.config(text='')  
        
        show_Detect_button(file_path)
    except Exception as e:
        print(f"Error: {e}")  # Print error for debugging


# Upload Button
upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)

# Place widgets
sign_image.pack(side='bottom', expand=True)
label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)

heading = Label(top, text="Age and Gender Detector", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()
