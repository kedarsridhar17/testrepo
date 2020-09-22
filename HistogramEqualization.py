import numpy as np
import scipy.misc, math
from PIL import Image

img = Image.open('images/lena512.bmp')

img1 = scipy.misc.fromimage(img)#N-D array conversion
fl = img1.flatten()#flatten the 2-D array into a 1-D array, histogram function accepts only 1-D array
hist, bins = np.histogram(img1,256,[0,255])

cdf = hist.cumsum()#cummalative distribution function
cdf_m = np.ma.masked_equal(cdf,0)#when cdf = 0, ignore and store other values in a new variable

num_cdf_m = (cdf_m - cdf_m.min())*255
den_cdf_m  = (cdf_m.max()-cdf_m.min())

cdf_m  = num_cdf_m / den_cdf_m

cdf = np.ma.filled(cdf_m,0).astype('uint8')#make sure that the masked places are zero 

im2 = cdf[fl]#this is a 1-D array

im3 = np.reshape(im2,img1.shape)#converting it to a 2-D array - giving im2 the shape of im1

im4 = scipy.misc.toimage(im3)#convert it into an image 

im4.show()
