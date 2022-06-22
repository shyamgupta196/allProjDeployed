
import streamlit as st
import torch.nn as nn 
import torch
import torchvision.transforms as T
from PIL import Image 
import os 
import torchvision

model = torchvision.models.alexnet(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Add a avgpool here
avgpool = nn.AdaptiveAvgPool2d((7, 7))

# Replace the classifier layer
# to customise it according to our output
model.classifier = nn.Sequential(
    nn.Linear(256 * 7 * 7, 1024),
    nn.Linear(1024, 256),
    nn.Linear(256, 2),
)



st.markdown(
    """
    <style>
    .reportview-container {
        background: black
    }
   .sidebar .sidebar-content {
        background: gray
    }
    </style>
    """,
    unsafe_allow_html=True,
)


uploadedfile = st.file_uploader("Upload an Image of Cat or Dog", ["png", "jpg", "jpeg"])

if not uploadedfile:
    st.warning("Please Upload a File with 'png','jpg','jpeg' extention ")
    st.stop()

def reader(name):
    img = Image.open(name)
    return img

if not os.path.exists('uploads'):
    os.mkdir('uploads')

PATH = 'uploads'

with open(os.path.join(PATH, uploadedfile.name), "wb") as f:
    f.write(uploadedfile.getbuffer())
        
image = reader(f"{PATH}/{uploadedfile.name}")
device = 'cpu'


checkpoint = torch.load(
    "ImprovedCatVsDogsModel.pth", map_location=torch.device("cpu")
)

model.load_state_dict(checkpoint['state_dict'])
model.to(device)

with torch.no_grad():
    transform = T.Compose(
        [T.Resize((128, 128)), T.ToTensor()]
    )
    tensor = transform(image)
    test_image_tensor = image
    x = model.features(tensor)
    x = avgpool(x)
    x = x.view(-1, 256 * 7 * 7)
    out = model.classifier(x)

MAP = {0: "Cat", 1: "Dog"}
st.header(f'Its a {MAP[out.numpy().argmax()]}')
