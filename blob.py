import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

#flatten to make greyscale, using your second red-black image as input.
file= open("count_micro.txt","a")
for i in range(45):
	if i<9:
		file_name="m_im000"+str(i+1)+".jpg"
	else:
		file_name="m_im00"+str(i+1)+".jpg"
	im = scipy.misc.imread("microaneurysms/"+file_name,flatten=1)
	#smooth and threshold as image has compression artifacts (jpg)
	im = ndimage.gaussian_filter(im, 2)
	im[im<10]=0
	blobs, number_of_blobs = ndimage.label(im)
	file.write(str(number_of_blobs)+",")
file.close()