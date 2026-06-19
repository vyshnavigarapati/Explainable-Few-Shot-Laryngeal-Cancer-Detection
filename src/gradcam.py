import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD MODEL
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 3)
)

model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()

# TARGET LAYER
target_layer = model.layer4[1].conv2

gradients = None
activations = None

# HOOKS
def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

classes = ["cancer", "normal", "precancer"]

def generate_gradcam(image_path):

    image = Image.open(image_path).convert("RGB")
    img = np.array(image)

    input_tensor = transform(image).unsqueeze(0).to(device)

    output = model(input_tensor)

    pred_class = output.argmax(dim=1)

    model.zero_grad()

    output[0, pred_class].backward()

    grads = gradients[0].cpu().data.numpy()
    acts = activations[0].cpu().data.numpy()

    weights = np.mean(grads, axis=(1,2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)

    cam = cv2.resize(cam, (224,224))

    cam = cam - np.min(cam)

    cam = cam / np.max(cam)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    original = cv2.resize(img, (224,224))

    superimposed = heatmap * 0.4 + original

    output_path = "models/gradcam_output.jpg"

    cv2.imwrite(output_path, superimposed)

    return output_path