import torch
import torchvision

def build_model(model_name, num_classes):
    
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            
        for param in model.parameters():
            param.requires_grad = False
    
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

        backbone = torch.nn.Sequential(*list(model.children())[:-1])
        head = model.fc
    
        return backbone, head, model
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def load_model(model_name, num_classes, checkpoint_path=None):
    backbone, head, model = build_model(model_name, num_classes)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
    return backbone, head, model
    
    
