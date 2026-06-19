import torch
from torchvision import transforms, models
from PIL import Image
from torch import nn

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= CLASSES =================
classes = ["cancer", "normal", "precancer"]

# ================= MODEL =================
model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 3)
)

model.load_state_dict(torch.load("models/best_model.pth"))

model = model.to(device)

model.eval()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ================= IMAGE =================
image_path = "test.jpg"

image = Image.open(image_path).convert("RGB")

image = transform(image).unsqueeze(0)

image = image.to(device)

# ================= PREDICTION =================
with torch.no_grad():

    output = model(image)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    confidence, predicted = torch.max(probabilities, 0)

print(f"Prediction: {classes[predicted.item()]}")
print(f"Confidence: {confidence.item()*100:.2f}%")