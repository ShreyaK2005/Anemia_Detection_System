import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import argparse

def load_model(model_path: str):
    """
    Load the ResNet18 model with trained weights.
    """
    model = models.resnet18(weights=None)  # No pretrained weights
    model.fc = nn.Linear(model.fc.in_features, 2)  # Match training architecture
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def register_hooks(target_layer, gradients, activations):
    """
    Register forward and backward hooks to capture activations and gradients.
    """
    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

def preprocess_image(image_path):
    """
    Load and preprocess the image.
    """
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor, img

def generate_gradcam(model, input_tensor, target_layer):
    """
    Generate Grad-CAM heatmap.
    """
    gradients = []
    activations = []

    register_hooks(target_layer, gradients, activations)

    # Forward pass
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Get gradients and activations
    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]

    # Compute weights
    weights = np.mean(grads, axis=(1, 2))

    # Generate CAM
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    return cam

def overlay_heatmap(img, cam, output_path):
    """
    Overlay heatmap on the original image and save.
    """
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, superimposed)
    print(f"GradCAM saved at: {output_path}")

def main(args):
    model = load_model(args.model_path)

    # Get target layer
    target_layer = model.layer4[-1]

    # Preprocess image
    input_tensor, original_img = preprocess_image(args.image_path)

    # Generate Grad-CAM
    cam = generate_gradcam(model, input_tensor, target_layer)

    # Save overlay
    overlay_heatmap(original_img, cam, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM for Anemia Detection")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights (.pth)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, default="gradcam_result.jpg", help="Path to save the output image")

    args = parser.parse_args()
    main(args)
