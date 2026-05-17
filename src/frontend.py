"""
frontend.py
===========
Tkinter-based GUI for the Anemia Screening System.

Workflow:
    1. User selects symptoms from the checklist.
    2. User uploads up to MAX_IMAGES eye images (or captures via webcam).
    3. Each image is auto-cropped to the palpebral conjunctiva, passed through
       a fine-tuned ResNet-18, and a Grad-CAM heatmap is displayed.
    4. After all images are processed, image-model scores are fused with the
       symptom score to produce a final Anemia / Normal prediction.

Dependencies:
    pip install torch torchvision pillow opencv-python numpy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageTk
from torchvision import models, transforms

from crop_utils import crop_conjunctiva

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_IMAGES = 4
IMG_SIZE = 224
ANEMIA_THRESHOLD = 0.8      # prob above this → confident Anemia
NORMAL_THRESHOLD = 0.5      # prob above this → confident Normal
IMAGE_WEIGHT = 0.6          # weight of model score in final fusion
SYMPTOM_WEIGHT = 0.4        # weight of symptom score in final fusion

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BG_COLOR    = "#ffe6f0"
PANEL_COLOR = "#fff0f5"
ACCENT      = "#cc0066"
BTN_PRIMARY = "#ff99cc"
BTN_SECONDARY = "#ffb3d9"
BTN_DANGER  = "#ff6666"

FONT        = ("Times New Roman", 12)
TITLE_FONT  = ("Times New Roman", 18, "bold")
SMALL_FONT  = ("Times New Roman", 10)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(weights_path: str = "best_model_final.pth") -> nn.Module:
    """Load a fine-tuned ResNet-18 binary classifier from disk."""
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.eval()
    return net


model = load_model()

# ---------------------------------------------------------------------------
# Image pre-processing pipeline
# ---------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Grad-CAM hooks
# ---------------------------------------------------------------------------

_gradients: list  = []
_activations: list = []
_target_layer = model.layer4[-1]


def _forward_hook(module, input, output):
    _activations.append(output)


def _backward_hook(module, grad_input, grad_output):
    _gradients.append(grad_output[0])


_target_layer.register_forward_hook(_forward_hook)
_target_layer.register_full_backward_hook(_backward_hook)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

anemia_probs: list = []
image_count: int   = 0

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def process_image(path: str):
    """
    Run inference and Grad-CAM on a single eye image.

    Parameters
    ----------
    path : str
        Filesystem path to the input image.

    Returns
    -------
    heatmap_overlay : np.ndarray
        BGR image with Grad-CAM heatmap blended over the cropped input.
    label : str
        "Anemia", "Normal", or "Uncertain".
    confidence : float
        Probability of the predicted class.
    anemia_prob : float
        Raw anemia probability (used for multi-image aggregation).
    """
    _gradients.clear()
    _activations.clear()

    img_cv   = cv2.imread(path)
    img_cv   = crop_conjunctiva(img_cv)
    img_pil  = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    input_tensor = transform(img_pil).unsqueeze(0)
    output       = model(input_tensor)
    probs        = torch.softmax(output, dim=1)

    anemia_prob = probs[0][0].item()
    normal_prob = probs[0][1].item()

    if anemia_prob > ANEMIA_THRESHOLD:
        label, confidence, pred_index = "Anemia", anemia_prob, 0
    elif normal_prob > NORMAL_THRESHOLD:
        label, confidence, pred_index = "Normal", normal_prob, 1
    else:
        pred_index  = torch.argmax(output, dim=1).item()
        label       = "Uncertain"
        confidence  = max(anemia_prob, normal_prob)

    # Backprop for Grad-CAM
    model.zero_grad()
    output[0, pred_index].backward()

    grads   = _gradients[0].detach().numpy()[0]    # (C, H, W)
    acts    = _activations[0].detach().numpy()[0]  # (C, H, W)
    weights = np.mean(grads, axis=(1, 2))          # (C,)

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for w, act_map in zip(weights, acts):
        cam += w * act_map

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    return overlay, label, confidence, anemia_prob


def get_final_prediction():
    """
    Fuse model scores across all uploaded images with the symptom score.

    Returns
    -------
    final_label : str
    final_confidence : float
    anemia_fused : float
    normal_fused : float
    """
    avg_anemia = np.mean(anemia_probs)

    checked_symptoms = sum(
        v.get() for k, v in symptoms.items() if k != "None of the Above"
    )
    total_symptoms = len(symptoms) - 1  # exclude "None of the Above"

    if symptoms["None of the Above"].get():
        symptom_score = 0.0
    else:
        symptom_score = checked_symptoms / total_symptoms

    anemia_fused = float(
        np.clip(avg_anemia * IMAGE_WEIGHT + symptom_score * SYMPTOM_WEIGHT, 0, 1)
    )
    normal_fused = 1.0 - anemia_fused

    if anemia_fused > normal_fused:
        return "Anemia", anemia_fused, anemia_fused, normal_fused
    return "Normal", normal_fused, anemia_fused, normal_fused


def capture_from_camera() -> str | None:
    """
    Open the default webcam and let the user capture a still frame.

    Returns
    -------
    str or None
        Path to the saved JPEG, or None if the user quit without capturing.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera.")
        return None

    messagebox.showinfo("Camera", "Press 's' to capture  |  'q' to cancel")
    saved_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera — press 's' to capture", frame)
        key = cv2.waitKey(1)
        if key == ord("s"):
            saved_path = f"captured_{image_count}.jpg"
            cv2.imwrite(saved_path, frame)
            break
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved_path

# ---------------------------------------------------------------------------
# GUI helpers
# ---------------------------------------------------------------------------

def _pil_to_tk(img_cv: np.ndarray, size: tuple[int, int] = (200, 200)) -> ImageTk.PhotoImage:
    """Convert a BGR OpenCV image to a Tkinter-compatible PhotoImage."""
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)).resize(size)
    return ImageTk.PhotoImage(img_pil)


def display_result(path: str) -> None:
    """Process one image and refresh all GUI widgets."""
    global image_count

    # Show cropped original
    cropped    = crop_conjunctiva(cv2.imread(path))
    tk_orig    = _pil_to_tk(cropped)
    original_label.config(image=tk_orig)
    original_label.image = tk_orig

    # Run model
    overlay, prediction, confidence, anemia_prob = process_image(path)
    anemia_probs.append(anemia_prob)
    image_count += 1

    # Show Grad-CAM overlay
    tk_heat = _pil_to_tk(overlay)
    heatmap_label.config(image=tk_heat)
    heatmap_label.image = tk_heat

    result_label.config(
        text=f"Image {image_count}: {prediction} ({confidence * 100:.1f}%)"
    )

    if image_count == 1:
        upload_btn.config(state="disabled")
        another_btn.config(state="normal")

    if image_count >= MAX_IMAGES:
        _show_final_results()


def _show_final_results() -> None:
    """Aggregate predictions and update the progress bars and result label."""
    final_label, final_conf, anemia_fused, normal_fused = get_final_prediction()

    anemia_pct = anemia_fused * 100
    normal_pct = normal_fused * 100

    anemia_bar["value"] = anemia_pct
    normal_bar["value"] = normal_pct
    anemia_pct_label.config(text=f"{anemia_pct:.1f}%")
    normal_pct_label.config(text=f"{normal_pct:.1f}%")

    result_label.config(
        text=(
            f"Final Prediction: {final_label} ({final_conf * 100:.1f}%)\n\n"
            f"    Anemia Risk:          {anemia_pct:.1f}%\n"
            f"    Normal Confidence:  {normal_pct:.1f}%"
        )
    )
    another_btn.config(state="disabled")


def choose_input() -> None:
    """Validate symptom selection, then route to file picker or webcam."""
    if sum(v.get() for v in symptoms.values()) == 0:
        messagebox.showwarning(
            "Selection Required",
            "Please select at least one symptom or 'None of the Above' before uploading.",
        )
        return

    use_file = messagebox.askquestion(
        "Image Source",
        "Choose image source:\n  Yes → File browser\n  No  → Webcam",
    )
    path = (
        filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if use_file == "yes"
        else capture_from_camera()
    )
    if path:
        display_result(path)


def reset_system() -> None:
    """Reset all state and GUI widgets for a fresh screening session."""
    global anemia_probs, image_count
    anemia_probs = []
    image_count  = 0

    original_label.config(image="")
    heatmap_label.config(image="")
    result_label.config(text="")

    upload_btn.config(state="normal")
    another_btn.config(state="disabled")

    anemia_bar["value"] = 0
    normal_bar["value"]  = 0
    anemia_pct_label.config(text="0%")
    normal_pct_label.config(text="0%")


def handle_none_selected() -> None:
    """Deselect all other symptoms when 'None of the Above' is ticked."""
    if symptoms["None of the Above"].get():
        for key in symptoms:
            if key != "None of the Above":
                symptoms[key].set(0)


def handle_symptom_selected() -> None:
    """Deselect 'None of the Above' when any specific symptom is ticked."""
    if any(symptoms[k].get() for k in symptoms if k != "None of the Above"):
        symptoms["None of the Above"].set(0)

# ---------------------------------------------------------------------------
# GUI layout
# ---------------------------------------------------------------------------

root = tk.Tk()
root.title("Anemia Screening System")
root.geometry("900x600")
root.configure(bg=BG_COLOR)

left_frame  = tk.Frame(root, bg=BG_COLOR)
left_frame.pack(side="left", fill="y", padx=20, pady=20)

right_frame = tk.Frame(root, bg=BG_COLOR)
right_frame.pack(side="right", expand=True, padx=20, pady=20)

# -- Symptom checklist -------------------------------------------------------

symptoms: dict[str, tk.IntVar] = {
    "Fatigue":              tk.IntVar(),
    "Pale Skin":            tk.IntVar(),
    "Shortness of Breath":  tk.IntVar(),
    "Dizziness":            tk.IntVar(),
    "Cold Hands/Feet":      tk.IntVar(),
    "None of the Above":    tk.IntVar(),
}

tk.Label(
    left_frame, text="Upload Eye Images (4)", font=TITLE_FONT, bg=BG_COLOR
).pack(pady=10)

for symptom_name, var in symptoms.items():
    command = (
        handle_none_selected
        if symptom_name == "None of the Above"
        else handle_symptom_selected
    )
    tk.Checkbutton(
        left_frame, text=symptom_name, variable=var,
        command=command, font=FONT, bg=BG_COLOR,
    ).pack(anchor="w")

# -- Action buttons ----------------------------------------------------------

upload_btn = tk.Button(
    left_frame, text="Upload Image",
    command=choose_input, font=FONT, bg=BTN_PRIMARY,
)
upload_btn.pack(pady=10)

another_btn = tk.Button(
    left_frame, text="Upload Another Image",
    command=choose_input, font=FONT, bg=BTN_SECONDARY, state="disabled",
)
another_btn.pack(pady=10)

# -- Image guidelines panel --------------------------------------------------

guidelines_frame = tk.Frame(left_frame, bg=PANEL_COLOR, bd=2, relief="ridge")
guidelines_frame.pack(pady=15, fill="x")

tk.Label(
    guidelines_frame, text="Image Guidelines",
    font=("Times New Roman", 13, "bold"), bg=PANEL_COLOR, fg=ACCENT,
).pack(anchor="w", padx=10, pady=(5, 5))

_good = [
    "✔ Ensure you are in a well-lit room",
    "✔ Focus on the lower pink region (palpebral conjunctiva)",
    "✔ Upload 4 images from different angles for best accuracy",
]
_warn = [
    "⚠ Avoid eyelashes covering the inner eyelid",
    "⚠ Avoid blurry or zoomed-out images",
]

for tip in _good:
    tk.Label(guidelines_frame, text=tip, font=FONT, bg=PANEL_COLOR, fg="green").pack(anchor="w", padx=15)

for tip in _warn:
    tk.Label(guidelines_frame, text=tip, font=FONT, bg=PANEL_COLOR, fg="red").pack(anchor="w", padx=15)

tk.Label(
    guidelines_frame,
    text="Gently pull down your lower eyelid while capturing",
    font=FONT, bg=PANEL_COLOR, fg="#333333",
).pack(anchor="w", padx=15, pady=(0, 5))

# -- Result display (right panel) --------------------------------------------

original_label = tk.Label(right_frame, bg=BG_COLOR)
original_label.pack()

heatmap_label = tk.Label(right_frame, bg=BG_COLOR)
heatmap_label.pack()

result_label = tk.Label(right_frame, font=FONT, bg=BG_COLOR, justify="left")
result_label.pack(pady=10)

# -- Confidence bars ---------------------------------------------------------

bar_frame = tk.Frame(right_frame, bg=BG_COLOR)
bar_frame.pack(pady=10)

style = ttk.Style()
style.theme_use("default")
style.configure("red.Horizontal.TProgressbar",   foreground="red",   background="red")
style.configure("green.Horizontal.TProgressbar", foreground="green", background="green")

tk.Label(bar_frame, text="Anemia Risk", font=FONT, bg=BG_COLOR).grid(row=0, column=0, sticky="w")
anemia_bar = ttk.Progressbar(bar_frame, length=200, maximum=100, style="red.Horizontal.TProgressbar")
anemia_bar.grid(row=0, column=1, padx=10)
anemia_pct_label = tk.Label(bar_frame, text="0%", bg=BG_COLOR)
anemia_pct_label.grid(row=0, column=2)

tk.Label(bar_frame, text="Normal", font=FONT, bg=BG_COLOR).grid(row=1, column=0, sticky="w")
normal_bar = ttk.Progressbar(bar_frame, length=200, maximum=100, style="green.Horizontal.TProgressbar")
normal_bar.grid(row=1, column=1, padx=10)
normal_pct_label = tk.Label(bar_frame, text="0%", bg=BG_COLOR)
normal_pct_label.grid(row=1, column=2)

# -- Utility buttons ---------------------------------------------------------

tk.Button(
    right_frame, text="Refresh",
    command=reset_system, font=FONT, bg=BTN_DANGER,
).pack(pady=10)

tk.Label(
    right_frame,
    text="* This is a screening tool, not a medical diagnosis.",
    fg="red", bg=BG_COLOR, font=SMALL_FONT,
).pack()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root.mainloop()
