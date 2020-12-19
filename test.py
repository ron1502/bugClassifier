import torchvision
import torch
import os

savePath = os.path.join("res", "model", "classificationResnet50")

print("Loading Model")
model = torchvision.models.resnet50(pretrained=False)
model.load_state_dict(torch.load(savePath))

model.eval()
