import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from model import AgeEstimator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AgeEstimator().to(device)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

def predict_age(image_path):
    image = Image.open(image_path).convert('RGB')
    img_t = transform(image).unsqueeze(0).to(device) 
    with torch.no_grad():
        output = model(img_t)

        pred_class = torch.argmax(output, dim=1).item()
    return pred_class

root = tk.Tk()
root.title("Age Estimation")

def load_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    img = Image.open(file_path)
    img.thumbnail((300,300))
    imgtk = ImageTk.PhotoImage(img)
    img_label.config(image=imgtk)
    img_label.image = imgtk

    try:
        age_class = predict_age(file_path)
        result_label.config(text=f"Predicted age class: {age_class}")
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")

btn = tk.Button(root, text="Load Image and Predict Age", command=load_and_predict)
btn.pack(pady=10)

img_label = tk.Label(root)
img_label.pack()

result_label = tk.Label(root, text="No image loaded", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
