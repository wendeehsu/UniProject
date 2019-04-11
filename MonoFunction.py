from PIL import Image, ImageEnhance
import numpy as np

def Monocrome(imgFile):
    """
    imput: 
        imgFile is the image's file name
        ex. "cat.jpg", "../cat.jpg"
    return:
        a numpy matrix with 1(True) for black, 0(False) for white
    """
    img = Image.open(imgFile)

    # resize image
    basewidth = 300
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    
    # Monocrome
    modifyImg = ImageEnhance.Contrast(img)
    newImg = modifyImg.enhance(5)
    grayImg = newImg.convert('1', dither=Image.NONE)
    
    return np.asmatrix(np.array(grayImg))

# print(Monocrome("Image/people.jpg"))