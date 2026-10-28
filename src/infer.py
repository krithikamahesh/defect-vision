import torch
import logging
import cv2
import os
import yaml

from data import build_transforms
from models import load_model

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
logging.basicConfig(level=logging.INFO)



def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = build_transforms(config['data']['transforms']['test'])(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

def preprocess_image_dir(image_dir):
    from torchvision import datasets
    input_dataset = datasets.ImageFolder(image_dir, transform=build_transforms(config['data']['transforms']['test']))
    input_dataloader = torch.utils.data.DataLoader(input_dataset, batch_size=1, shuffle=False)
    return input_dataloader


def infer(model, input):

    if isinstance(input, torch.utils.data.DataLoader):
        probs = []
        for image_batch, _ in input:
            model.eval()
            with torch.no_grad():
                output = model(image_batch)
                prob = torch.softmax(output, dim=1)
                probs.append(prob)
        return torch.cat(probs, dim=0)
    else:
        model.eval()
        with torch.no_grad():
            output = model(input)
            prob = torch.softmax(output, dim=1)
        return prob

def run_inference(model, data):
    model = load_model(config['model']['name'], config['model']['num_classes'], config['model']['checkpoint_path'])[2]
    if data is None:
        logging.error("No data provided for inference")
        return None
    elif os.path.isdir(data):
        input_dataloader = preprocess_image_dir(data)
        probs = infer(model, input_dataloader)
    else:
        input_tensor = preprocess_image(data)
        probs = infer(model, input_tensor)

    logging.info("Class predicted: %s", torch.argmax(probs, dim=1))
    return probs
