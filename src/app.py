import streamlit as st
import torch

from PIL import Image
from torchvision import transforms, models
from torch import nn
from gradcam import generate_gradcam

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= MODEL =================
model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 3)
)

model.load_state_dict(torch.load("models/best_model.pth"))

model.to(device)

model.eval()

classes = ["cancer", "normal", "precancer"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ================= STREAMLIT UI =================
st.title("Explainable Few-Shot Laryngeal Cancer Detection")

uploaded_file = st.file_uploader(
    "Upload Endoscopic Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        confidence, predicted = torch.max(probabilities, 0)

    prediction = classes[predicted.item()]

    st.success(f"Prediction: {prediction}")

    st.info(f"Confidence: {confidence.item()*100:.2f}%")

    # SAVE IMAGE
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # GENERATE GRADCAM
    gradcam_path = generate_gradcam("temp.jpg")

    # SHOW HEATMAP
    st.image(gradcam_path, caption="Grad-CAM Heatmap")