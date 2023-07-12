import os
import io
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# Define the model right in this script
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, 5)

    def forward(self, x):
        return self.base_model(x)

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomNet()
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    return model.to(device)

def input_fn(request_body, request_content_type='application/x-image'):
    if request_content_type == 'application/x-image':
        image = Image.open(io.BytesIO(request_body))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_data = input_data.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()
