import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torch
from os import listdir
from PIL import Image
import pandas as pd

def init_resnet(seed):
    '''Initialization of the resNet: we exclude last layer because it is for classification and instead we use it to extract features from the images, i.e. to get useful information from the image pixels before dimensionality reduction. Model set to eval mode
    seed: int value defining random state in order to have reproducibility.'''
    torch.manual_seed(seed)
    model = models.resnet18(pretrained=True)
    newModel = nn.Sequential(*list(model.children())[:-1])      #exclude last layer that is for classification
    newModel.eval()
    return newModel

def img_transformation(img):
    '''Img conversion to RGB and resize to have a shape 224*224 needed as input to the resnet
    img: PIL image we want to transform'''
    img3d = img.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    transformed_img = transform(img3d)
    final_img = transformed_img.unsqueeze(0)        # add one dim needed to run a resnet on it
    return final_img

def feature_selection_from_img(img, model):
    '''Takes in input img and model, transform img, get the tensor, apply the model to tensor extracting useful features from pixels,
     transform the output of the net in an array to be used in a dataframe used as input for the governance pipeline
     img: PIL image we want to do feature selection on
     model: pre-trained resnet18 model, already initialized'''
    tensor_img = img_transformation(img)
    with torch.no_grad():
        feature = model(tensor_img)     #feature is tensor of size [1, 512, 1, 1]
    rfeat = feature.reshape(512,1)
    arr = rfeat.numpy().reshape(-1)

    return arr

def df_from_folder(folder_path, model, desired_list=None):
    '''Take in input the path of the folder which contains the images. 
    Returns a dataframe where each row is an image in numpy mode- 
    If desired_list is not None it takes only that images in the folder
    folder_path: directory where the images are
    model:  pre-trained resnet18 model, already initialized
    desired_list: list of names of images we want to pre-process. If None all folder_path is taken.'''
    if (desired_list==None):
        imagesList = listdir(folder_path)
    else:
        imagesList = desired_list
        
    arrayList = []
    for image_path in imagesList:
        img = Image.open(folder_path+image_path)
        # apply model and get array
        arr = feature_selection_from_img(img,model)       
        arrayList.append(arr)

    # A dataframe is needed in order to re-apply the same procedure for governance pipeline defined on tabular data    
    df = pd.DataFrame(arrayList)
    return df