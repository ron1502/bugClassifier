import torchvision.models as models
import torch.nn as nn
from bugDataset import *
from collections import OrderedDict
import cv2
import numpy as np
from PIL import Image
'''
Extracting CAM
--Changes in training and testing
x 1. Modify output channels to match class count of conv3 in layer4 section 3
x 2. Modify input size of batch normalization in in layer4 section 3
x 3. Adaptive avg pooling layer already generates and output of 1x1/input channel, no changes needed
x 4. Change fully connected layer to  [classCount, classCount] and set bias False.
     Ignore Softmax layer (Layer not need for computation)
--changes in testing
5. Obtain weights fc
6. Register hook to extract activation maps of covn3/layer4/sec3
7. Compute and display CAM
'''

'''
Visualizing CAM
OpenCV pocesses and addweighted function that performs superposition.
Look into it when map obtained
'''

reluActivationMap = {}

def getPrepedResnet50():
    classCount = len(indexToLabel)
    model = models.resnet50(pretrained=False)
    model.layer4[2].conv3 = nn.Conv2d(512, classCount, kernel_size=(1,1), stride=(1, 1), bias=False)
    model.layer4[2].bn3 = nn.BatchNorm2d(classCount)
    model.layer4[2].downsample = nn.Sequential(OrderedDict([('conv4', nn.Conv2d(2048, classCount, (1, 1))),
                                                            ('bn4', nn.BatchNorm2d(classCount))]))
    model.fc = nn.Linear(classCount, classCount, bias=False)
    return model

#Fixe dimensions to perform addWeighted
def displayCAM(model, orgImg, classIndex):
    cam = computeCam(model, classIndex)
    print(cam)
    cam = (cam * 255/np.max(cam)).astype(np.uint8)
    print(cam)
    cam = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    orgImg = cv2.cvtColor(np.array(orgImg), cv2.COLOR_RGB2BGR)

    height, width, _ = orgImg.shape
    cam = cv2.resize(cam, (width, height))


    print("camSize: {}".format(cam.shape))
    print("imgSize: {}".format(orgImg.shape))

    img = cv2.addWeighted(cam, 0.5, orgImg, 0.5, 0)
    cv2.imshow("Heat map", img)
    cv2.waitKey()

def hookGetReLuOutput(module, inGrad, outGrad):
    reluActivationMap["layer4"] = outGrad.data

def setupModel(model):
    model.layer4[2].register_forward_hook(hookGetReLuOutput)
    return

def computeCam(model, classIndex, test=False):
    print("Maps: {}".format(reluActivationMap["layer4"].size()))
    maps = torch.squeeze(reluActivationMap["layer4"], 0)
    weights = model.fc.weight[:] [classIndex].unsqueeze(1).unsqueeze(1)
    cams = torch.mul(maps, weights)
    if(test):
        print("Weight Dims: {}".format(weights.size()))
        print("Map Dims: {}".format(maps.size()))
        print("Cams size: {}".format(cams.size()))
    return torch.sum(cams, 0).detach().cpu().numpy()



if __name__ == "__main__":
    print("PrepResnet50")
    model = getPrepedResnet50()
    setupModel(model)

    modelPath = os.path.join("res", "model", "CAM-resnet50-447")

    model.load_state_dict(torch.load(modelPath))

    model.eval()

    prepImg = trans.Compose([
        trans.ToTensor(),
        trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    datasetPath = os.path.join("8-2Dataset", "test")
    imgFolder = "104ANASA"
    imagePath = os.path.join(imgFolder, "DSC_0076.JPG")
    bugClass = indexToLabel.index(folderToName[imgFolder])

    orgImg = Image.open(os.path.join(datasetPath, imagePath)).resize((376, 250))
    testImg = prepImg(orgImg).unsqueeze(0)

    output = model(testImg)

    inf = torch.argmax(output).item()
    print("Class: {} - Inference: {}".format(bugClass, inf))

    displayCAM(model, orgImg.convert('RGB'), inf)
