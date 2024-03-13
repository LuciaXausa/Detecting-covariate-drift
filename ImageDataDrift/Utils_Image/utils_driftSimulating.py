import os
from PIL import ImageDraw,  ImageEnhance, Image
import numpy as np


def from_npg_to_jpg(input_path, output_path):
    '''Convert images in input folder that are saved in npg to images 
    in jpg and save them in output folder
    input_path: directory of images we want to convert to jpg
    output_path: directory where we want to save the jpg converted images'''
    imageList = os.listdir(input_path)
    for image_path in imageList:
        img_png = Image.open(input_path+image_path) 
        img_png.save(output_path+image_path+'.jpg')


def black_filter(image):
    '''Cover in black the center and the background
    image: PIL image we want to apply the black filter to.'''

    # Create a mask with a white circle in the center of the images to get implicitly the background
    mask_im = Image.new('L', image.size)
    draw = ImageDraw.Draw(mask_im)
    draw.ellipse((150,60, 690, 600), fill = 255)
    mask_im.save('mask_circle.jpg', quality = 95)
    
    #create background image
    back_im = Image.new(mode = 'L' , size = (800,600))
    #paste image and mask
    back_im.paste(image, (0,0), mask_im)
    back_im.save('composed.jpg', quality = 95)

    # draw black circle centered
    draw = ImageDraw.Draw(back_im)
    draw.ellipse((250,150, 600, 500), fill = 'black')

    # cut the borders 
    area = (150,60, 690,600)
    cropped_img = back_im.crop(area)

    return cropped_img


def add_Gaussian_noise(image, seed=1, sigma=1):
    '''Add Gaussian noise to single image with sigma^2 equal to sigma
    image: PIL image we want to apply Gaussian noise on
    seed: numerical value, to set the random state for generating gaussian noise
    sigma: numerical value, is the variance of gaussian noise'''
    rng = np.random.RandomState(seed)
    np_frame = np.array(image)
    gaussian = rng.normal(0.0, sigma, (600, 800))  # np_frame.shape = (600, 800)
    np_drifted = np_frame + gaussian
    drifted_image = Image.fromarray(np_drifted)
    drifted_image = drifted_image.convert('RGB')        # I need to do this otherwise I cannot save it
    # it returned OSError: cannot write mode F as JPEG
    return drifted_image

def change_intensity_greys(image, shiftValue=40):
    '''Change intensity of greys in the image. 
    image: PIL image we want to chenge the greys to.
    shiftValue: numerical value defining how muche we are changing the colors.'''
    newImage = image.convert('L')       #colors in grey scale
    d = newImage.getdata()
    list_image = []
    for item in d:
        if item in list(range(60,180)):         #if the pixel is gray, empirically found
            list_image.append(item+shiftValue)          # reduce its intensity
        else:
            list_image.append(item)             # else maintain its value
    newImage.putdata(list_image)
    return newImage

# Apply on multi image:

def only_image_folder(mydir):
    '''used only once to get rid of file Zone.Identifier and get only images
    mydir: directory of folder we want to 'clean' keeping only image files'''
    for f in os.listdir(mydir):
       if not f.endswith('.jpg') or f.endswith('.png'):
            os.remove(os.path.join(mydir, f))

    
def create_black_folder(input_path, output_path):
    '''used to create folder with black centered circle
    input_path: directory of original images
    output_path: directory where we want to save filtered images.'''
    imagesList = os.listdir(input_path)
    for image_path in imagesList:
        img = Image.open(input_path+image_path)
        black_img = black_filter(img)
        black_img.save(output_path+image_path)


def create_gaussian_folder(input_path, output_path, sigma, seed=1):
    '''used to create folder with gaussian-drifted images
    input_path: directory of original images
    output_path: directory where we want to save synthethic images
    sigma: int value defining the variance of gaussian noise
    seed: int value that set the random state for simulating gaussian noise.'''
    imagesList = os.listdir(input_path)
    for image_path in imagesList:
        img = Image.open(input_path+image_path)
        drifted_img = add_Gaussian_noise(img, seed=seed, sigma=sigma)
        black_drifted_img = black_filter(drifted_img)
        black_drifted_img.save(output_path+image_path)


def create_intensity_folder(input_path, output_path, shiftValue = 40):
    '''used to create folder with images with changed grays intensity
    input_path: directory of original images
    output_path: directory where we want to save synthethic images
    shiftValue: numerical value defining how muche we are changing the colors.'''
    imagesList = os.listdir(input_path)
    for image_path in imagesList:
        img = Image.open(input_path+image_path)
        drifted_img = change_intensity_greys(img, shiftValue)
        black_drifted_img = black_filter(drifted_img)
        black_drifted_img.save(output_path+image_path)