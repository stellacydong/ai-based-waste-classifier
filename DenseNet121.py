import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# convert data to a normalized torch.FloatTensor
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

#defining classes
classes=['o','r']


# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
model.to(device);

# Loading the last saved model 
model.load_state_dict(torch.load('model_waste.pt', map_location=torch.device('cpu')))

# classify 
def classify(image):
    # transform the image 
    image_tensor = test_transforms(image)

    # create a mini-batch as expected by the model
    image_tensor_batch = image_tensor.unsqueeze(0) 

    model.eval()
    output = model(image_tensor_batch)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) 

    # compute the probability of the image being O or R class 
    o_prob,r_prob = probabilities


    #print the result 
    if classes[preds] == 'r': 
        result = 'Recyclable'
    else:
        result = 'Organic'
        
    statement = f'\nThe probability of the object being Organic is {o_prob.detach().numpy()*100:.3f}%.       \nThe probability of the object being Recyclable is {r_prob.detach().numpy()*100:.3f}%.       \nTherefore, the object is classified as ' + result + '.' 
    
    return statement

