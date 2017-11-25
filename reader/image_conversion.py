import numpy as np 
from scipy import misc
import matplotlib
import matplotlib.pyplot as plt
def readGrayScaleImage(filename):
	global img
	#img=np.array(misc.imread('./test_pics/8.jpg',flatten=True,mode='L'))
	img=np.array(misc.imread(filename,flatten=True,mode='L'))
	#print(a.shape)
	img=misc.imresize(img,(500,500),interp='cubic')
	#print(img.shape)
	

def displayImage():
	global img
	plt.imshow(img,cmap = matplotlib.cm.Greys_r)
	plt.show()

def applyThreshhold(th):
	# this function sets all the pixels values with intensity higher than "th" to 255 and lower to 0
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j]>th:
				img[i][j]=255
			else:
				img[i][j]=0

def removeEmptyRows():
	# This function is used to remove the white pages spaces above and below the written character in the image.
	# It ,by default, leaves 10 white page spaces above and below the written character.
	# Those extra 10 spaces are deliberately left out which will come handy while resizing the image.
	global img
	mark=-1
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j]==0:
				mark=i
				break
		if mark>=0:
			break
	mark=mark-10
	img=img[mark:img.shape[0]-1][:]
	#print(mark)
	#print(img.shape)
	mark=-1
	for i in range(img.shape[0]-1,0,-1):
		for j in range(img.shape[1]):
			if img[i][j]==0:
				mark=i
				break
		if mark>=0:
			break
	mark=mark+10
	img=img[0:mark][:]
	#print(mark)
	#print(img.shape)

def removeEmptyColumns():
	# Similar to removeEmptyRows(). It just removes columns.
	global img
	mark=-1
	for i in range(img.shape[1]):
		for j in range(img.shape[0]):
			if img[j][i]==0:
				mark=i
				break
		if mark>=0:
			break
	mark=mark-10
	#print(mark)
	img=img[:,mark:img.shape[1]-1]
	mark=-1
	for i in range(img.shape[1]-1,0,-1):
		for j in range(img.shape[0]):
			if img[j][i]==0:
				mark=i
				break
		if mark>=0:
			break
	mark=mark+10
	img=img[:,0:mark]

def increaseWidth():
	# Colors all the surrounding pixels of a pixel with the same color as the pixel.
	global img
	#print(img.shape)
	tempImg=np.copy(img)
	#print(tempImg.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j]==0:
				tempImg[i][j]=0
				tempImg[i+1][j]=0
				tempImg[i-1][j]=0
				tempImg[i][j-1]=0
				tempImg[i+1][j-1]=0
				tempImg[i-1][j-1]=0
				tempImg[i][j+1]=0
				tempImg[i+1][j+1]=0
				tempImg[i-1][j+1]=0
	img=tempImg[:,:]

def resizeImage():
	global img
	img=misc.imresize(img,(28,28),interp='cubic')

def invert():
	# this became necessary as leo considers 0 as white and 255 as black.
	global img
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j]==0:
				img[i][j]=1
			else:
				img[i][j]=0

def convertToRowMajor():
	# this function converts the complete two 2 dimensional image into a 1D array.
	global img
	x=np.zeros(784)
	k=0
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			x[k]=img[i][j]
			k+=1
	return x

def convert(filename):
	# This is the main function
	global img
	readGrayScaleImage(filename)
	applyThreshhold(100)
	removeEmptyRows()
	removeEmptyColumns()
	increaseWidth()
	resizeImage()
	applyThreshhold(200)
	invert()
	return convertToRowMajor()
	# displayImage()
# convert('./test_pics/8.jpg')