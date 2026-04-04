import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from crop_utils import crop_conjunctiva

# =========================
# LOAD MODEL
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("best_model_final.pth", map_location="cpu"))
model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# GRADCAM SETUP
# =========================
target_layer = model.layer4[-1]

gradients = []
activations = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# =========================
# MULTI IMAGE STORAGE
# =========================
anemia_probs = []
image_count = 0
MAX_IMAGES = 4

# =========================
# PROCESS IMAGE
# =========================
def process_image(path):
    gradients.clear()
    activations.clear()

    img_cv = cv2.imread(path)
    img_cv = crop_conjunctiva(img_cv)

    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0)

    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)

    anemia_prob = probabilities[0][0].item()
    normal_prob = probabilities[0][1].item()

    if anemia_prob > 0.8:
        label = "Anemia"
        confidence = anemia_prob
        pred_index = 0
    elif normal_prob > 0.5:
        label = "Normal"
        confidence = normal_prob
        pred_index = 1
    else:
        label = "Uncertain"
        confidence = max(anemia_prob, normal_prob)
        pred_index = torch.argmax(output, 1).item()

    model.zero_grad()
    output[0, pred_index].backward()

    grads = gradients[0].detach().numpy()[0]
    acts = activations[0].detach().numpy()[0]

    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    result = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    return result, label, confidence, anemia_prob

# =========================
# FINAL AGGREGATION
# =========================
def get_final_prediction():
    avg_anemia = np.mean(anemia_probs)

    # =========================
    # SYMPTOM SCORE
    # =========================
    selected = sum(v.get() for k,v in symptoms.items() if k != "None of the Above")
    total = len(symptoms) - 1  # exclude "None of the Above"

    if symptoms["None of the Above"].get():
        symptom_score = 0
    else:
        symptom_score = selected / total

    # =========================
    # COMBINED SCORE (TUNABLE)
    # =========================
    final_anemia = (avg_anemia * 0.6) + (symptom_score * 0.4)

    # Clamp just in case
    final_anemia = min(max(final_anemia, 0), 1)
    final_normal = 1 - final_anemia

    # =========================
    # FINAL LABEL
    # =========================
    if final_anemia > final_normal:
        final_prediction = "Anemia"
        final_confidence = final_anemia
    else:
        final_prediction = "Normal"
        final_confidence = final_normal

    return final_prediction, final_confidence, final_anemia, final_normal

# =========================
# CAMERA FUNCTION
# =========================
def capture_from_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera")
        return None

    messagebox.showinfo("Camera", "Press 's' to capture, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            path = f"captured_{image_count}.jpg"
            cv2.imwrite(path, frame)
            break
        elif key == ord('q'):
            path = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return path

# =========================
# GUI
# =========================
root = tk.Tk()
root.title("Anemia Screening System")
root.geometry("900x600")
root.configure(bg="#ffe6f0")

FONT = ("Times New Roman", 12)
TITLE_FONT = ("Times New Roman", 18, "bold")

left_frame = tk.Frame(root, bg="#ffe6f0")
left_frame.pack(side="left", fill="y", padx=20, pady=20)

right_frame = tk.Frame(root, bg="#ffe6f0")
right_frame.pack(side="right", expand=True, padx=20, pady=20)

# =========================
# SYMPTOMS
# =========================
symptoms = {
    "Fatigue": tk.IntVar(),
    "Pale Skin": tk.IntVar(),
    "Shortness of Breath": tk.IntVar(),
    "Dizziness": tk.IntVar(),
    "Cold Hands/Feet": tk.IntVar(),
    "None of the Above": tk.IntVar()
}

def handle_none():
    if symptoms["None of the Above"].get():
        for k in symptoms:
            if k != "None of the Above":
                symptoms[k].set(0)

def handle_other():
    if any(symptoms[k].get() for k in symptoms if k != "None of the Above"):
        symptoms["None of the Above"].set(0)

tk.Label(left_frame, text="Upload Eye Images (4)", font=TITLE_FONT, bg="#ffe6f0").pack(pady=10)

for s, v in symptoms.items():
    tk.Checkbutton(left_frame, text=s, variable=v,
                   command=handle_none if s=="None of the Above" else handle_other,
                   font=FONT, bg="#ffe6f0").pack(anchor='w')

original_label = tk.Label(right_frame, bg="#ffe6f0")
original_label.pack()

heatmap_label = tk.Label(right_frame, bg="#ffe6f0")
heatmap_label.pack()

result_label = tk.Label(right_frame, font=FONT, bg="#ffe6f0")
result_label.pack(pady=10)










# =========================
# CONFIDENCE BARS
# =========================
bar_frame = tk.Frame(right_frame, bg="#ffe6f0")
bar_frame.pack(pady=10)

# Anemia Bar
tk.Label(bar_frame, text="Anemia Risk", font=FONT, bg="#ffe6f0").grid(row=0, column=0, sticky="w")

anemia_bar = ttk.Progressbar(bar_frame, length=200, maximum=100)
anemia_bar.grid(row=0, column=1, padx=10)

anemia_percent_label = tk.Label(bar_frame, text="0%", bg="#ffe6f0")
anemia_percent_label.grid(row=0, column=2)

# Normal Bar
tk.Label(bar_frame, text="Normal", font=FONT, bg="#ffe6f0").grid(row=1, column=0, sticky="w")

normal_bar = ttk.Progressbar(bar_frame, length=200, maximum=100)
normal_bar.grid(row=1, column=1, padx=10)

normal_percent_label = tk.Label(bar_frame, text="0%", bg="#ffe6f0")
normal_percent_label.grid(row=1, column=2)

style = ttk.Style()
style.theme_use('default')

style.configure("red.Horizontal.TProgressbar", foreground='red', background='red')
style.configure("green.Horizontal.TProgressbar", foreground='green', background='green')

anemia_bar.config(style="red.Horizontal.TProgressbar")
normal_bar.config(style="green.Horizontal.TProgressbar")
















# =========================
# DISPLAY FUNCTION
# =========================
def display_result(path):
    global image_count

    img_cv = cv2.imread(path)
    cropped = crop_conjunctiva(img_cv)

    img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    img = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    original_label.config(image=img_tk)
    original_label.image = img_tk

    heatmap_img, prediction, confidence, anemia_prob = process_image(path)

    anemia_probs.append(anemia_prob)
    image_count += 1

    cv2.imwrite("temp.jpg", heatmap_img)
    heat = Image.open("temp.jpg").resize((200,200))
    heat_tk = ImageTk.PhotoImage(heat)

    heatmap_label.config(image=heat_tk)
    heatmap_label.image = heat_tk

    # Convert to percentage
    confidence_percent = confidence * 100

    result_label.config(
        text=f"Image {image_count}: {prediction} ({confidence_percent:.1f}%)"
    )

    if image_count == 1:
        upload_btn.config(state="disabled")
        another_btn.config(state="normal")

    if image_count >= MAX_IMAGES:
        final_prediction, final_confidence, anemia_prob_final, normal_prob_final = get_final_prediction()

        final_percent = final_confidence * 100
        anemia_percent = anemia_prob_final * 100
        normal_percent = normal_prob_final * 100
        anemia_bar["value"] = anemia_percent
        normal_bar["value"] = normal_percent

        anemia_percent_label.config(text=f"{anemia_percent:.1f}%")
        normal_percent_label.config(text=f"{normal_percent:.1f}%")

        result_label.config(
            text=f"""Final Prediction: {final_prediction} ({final_percent:.1f}%)

    Anemia Risk: {anemia_percent:.1f}%
    Normal Confidence: {normal_percent:.1f}%"""

        )

        another_btn.config(state="disabled")

# =========================
# INPUT HANDLER
# =========================
def choose_input(first=False):

    #  VALIDATION CHECK
    if sum(v.get() for v in symptoms.values()) == 0:
        messagebox.showwarning(
            "Selection Required",
            "Please select at least one symptom or 'None of the Above' before uploading."
        )
        return

    choice = messagebox.askquestion("Input","Yes=Computer\nNo=Camera")

    if choice=="yes":
        path = filedialog.askopenfilename(filetypes=[("Image","*.jpg *.png *.jpeg")])
    else:
        path = capture_from_camera()

    if path:
        display_result(path)

def reset_system():
    global anemia_probs, image_count

    anemia_probs = []
    image_count = 0

    original_label.config(image="")
    heatmap_label.config(image="")
    result_label.config(text="")

    upload_btn.config(state="normal")
    another_btn.config(state="disabled")
    anemia_bar["value"] = 0
    normal_bar["value"] = 0

    anemia_percent_label.config(text="0%")
    normal_percent_label.config(text="0%")
# =========================
# BUTTONS
# =========================
upload_btn = tk.Button(left_frame, text="Upload Image",
                       command=lambda: choose_input(True),
                       font=FONT, bg="#ff99cc")
upload_btn.pack(pady=10)

another_btn = tk.Button(left_frame, text="Upload Another Image",
                        command=lambda: choose_input(False),
                        font=FONT, bg="#ffb3d9",
                        state="disabled")
another_btn.pack(pady=10)





# =========================
# INSTRUCTIONS SECTION (ENHANCED)
# =========================
instructions_frame = tk.Frame(left_frame, bg="#fff0f5", bd=2, relief="ridge")
instructions_frame.pack(pady=15, fill="x")

tk.Label(
    instructions_frame,
    text=""
         ""
         ""
         ""
         ""
         "📌 Image Guidelines",
    font=("Times New Roman", 13, "bold"),
    bg="#fff0f5",
    fg="#cc0066"
).pack(anchor="w", padx=10, pady=(5,5))

# ✔ Good practices (Green)
tk.Label(
    instructions_frame,
    text="✔ Ensure you are in a well-lit room",
    font=FONT,
    bg="#fff0f5",
    fg="green"
).pack(anchor="w", padx=15)

tk.Label(
    instructions_frame,
    text="✔ Focus on the lower pink region (palpebral conjunctiva)",
    font=FONT,
    bg="#fff0f5",
    fg="green"
).pack(anchor="w", padx=15)

tk.Label(
    instructions_frame,
    text="✔ Upload 4 images from different angles for best accuracy",
    font=FONT,
    bg="#fff0f5",
    fg="green"
).pack(anchor="w", padx=15)

# ⚠ Warnings (Red)
tk.Label(
    instructions_frame,
    text="⚠ Avoid eyelashes covering the inner eyelid",
    font=FONT,
    bg="#fff0f5",
    fg="red"
).pack(anchor="w", padx=15)

tk.Label(
    instructions_frame,
    text="⚠ Avoid blurry or zoomed-out images",
    font=FONT,
    bg="#fff0f5",
    fg="red"
).pack(anchor="w", padx=15)

tk.Label(
    instructions_frame,
    text="📸 Gently pull down your lower eyelid while capturing",
    font=FONT,
    bg="#fff0f5",
    fg="#333333"
).pack(anchor="w", padx=15, pady=(0,5))








#  ALWAYS ENABLED REFRESH BUTTON
refresh_btn = tk.Button(right_frame, text="Refresh",
                        command=lambda: reset_system(),
                        font=FONT, bg="#ff6666")
refresh_btn.pack(pady=10)

# =========================
# BOTTOM LEFT DISCLAIMER (FIXED)
# =========================
tk.Label(right_frame, text="*This is a screening tool, not a medical diagnosis", fg="red", bg="#ffe6f0", font=("Times New Roman", 10)).pack()

root.mainloop()
