#!/usr/bin/env python
# coding: utf-8

# In[237]:


from PIL import Image, ImageEnhance, ImageFilter

import requests
from io import BytesIO
from IPython.display import Image as ImageDisplay

outputfile = "Image/modified.png"
cat = "Image/cat.jpg"
cake = "Image/cake.jpg"
woman = "Image/people.jpg"
man = "Image/man.jpg"


# In[244]:


img = Image.open(woman)

basewidth = 300
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), Image.ANTIALIAS)
img


# In[245]:


### contrast -> monochrome
modifyImg = ImageEnhance.Contrast(img)
newImg = modifyImg.enhance(5)
grayImg = newImg.convert('1', dither=Image.NONE)
grayImg


# In[246]:


### monochrome
img.convert('1', dither=Image.NONE)


# In[247]:


### blur -> contrast -> monochrome
testImg = img.filter(ImageFilter.BLUR)
modifyBlurImg = ImageEnhance.Contrast(testImg)
bwImg = modifyBlurImg.enhance(10).convert('1', dither=Image.NONE)
bwImg


# In[248]:


### monochrome -> blur
img.convert('1', dither=Image.NONE).filter(ImageFilter.BLUR)


# In[249]:


### blur -> contrast -> monochrome -> blur
bwImg.filter(ImageFilter.BLUR)

