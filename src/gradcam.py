import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# LOAD MODEL (FIXED)
# =========================
model = models.resnet18(weights=None)   # no pretrained weights

# IMPORTANT: match training architecture
model.fc = nn.Linear(model.fc.in_features, 2)

# Load trained weights
model.load_state_dict(torch.load("best_model_final.pth", map_location="cpu"))
model.eval()

# =========================
# TARGET LAYER
# =========================
target_layer = model.layer4[-1]

# =========================
# HOOKS (UPDATED)
# =========================
gradients = []
activations = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer.register_forward_hook(forward_hook)

# NEW (fixes PyTorch warning)
target_layer.register_full_backward_hook(backward_hook)

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD IMAGE
# =========================
img_path = "master_dataset_1/test/anemia/kaggle_anemia_42.png"

img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0)

# =========================
# FORWARD PASS
# =========================
output = model(input_tensor)
pred_class = output.argmax(dim=1)

# =========================
# BACKWARD PASS
# =========================
model.zero_grad()
output[0, pred_class].backward()

# =========================
# GRAD-CAM CALCULATION
# =========================
grads = gradients[0].detach().numpy()[0]
acts = activations[0].detach().numpy()[0]

weights = np.mean(grads, axis=(1,2))

cam = np.zeros(acts.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * acts[i]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224,224))

# Normalize
cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)

# =========================
# HEATMAP OVERLAY
# =========================
img_cv = cv2.imread(img_path)
img_cv = cv2.resize(img_cv, (224,224))

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

superimposed = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

# =========================
# DISPLAY
# =========================
# Save output instead of showing
output_path = "gradcam_result.jpg"
cv2.imwrite(output_path, superimposed)

print(f"GradCAM saved at: {output_path}")
