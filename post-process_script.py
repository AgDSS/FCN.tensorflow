# Import libraries
import cv2
import numpy as np
import scipy
from skimage import measure, io,feature
from skimage.morphology import reconstruction
from matplotlib import pyplot as plt
import functools
import os
from glob import glob

def main():
	# Set input, output, base, and overlay directory
	in_dir = 'C:/Users/there/anaconda3/envs/postprocess/images/input'
	out_dir = 'C:/Users/there/anaconda3/envs/postprocess/images/predictions'
	base_dir = 'C:/Users/there/anaconda3/envs/postprocess/images/base'
	overlay_dir = 'C:/Users/there/anaconda3/envs/postprocess/images/overlay'

	# Find image names in a directory
	img_fnames = get_image_fnames(in_dir, recursive=True)

	# Loop through each image
	for image in range(len(img_fnames)):
		name = '{0}/pred{1}.png'.format(out_dir, image+1)
		img = cv2.imread(img_fnames[image])
		img_processing(img, name)

# Function to read images from file
def get_image_fnames(directory, recursive=False):
	if recursive:
		return glob(os.path.join(directory, "**", "*.png"), recursive=True)
	else:
		return glob(os.path.join(directory, "*.png"), recursive=False)

# Switch BGR to RGB, output independent channels
def pre_process(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	r, g, b = cv2.split(img)
	return r,g,b

# Label regions in a 2D image
def label(channel):
	labels, num_labels = measure.label(channel, return_num=True)
	return labels, num_labels

# Extract region properties from a labeled 2D image
def properties(labelled_img):
	props = measure.regionprops(labelled_img)
	return props

# Delete random, noisy predicitions; determined by pixel area
def del_noise(img, labelled_img, num_labels, props, THRESHOLD=2000):
	img[functools.reduce(lambda x,y: x | y,
		[labelled_img[:,:] == x+1 for x in range(0,num_labels) if props[x].area < THRESHOLD],
		np.zeros(img.shape,dtype=bool))] = 0
	return img

# Fill in holes in an image
def fill_holes(img):
	seed = np.copy(img)
	seed[1:-1, 1:-1] = img.max()
	mask = img
	filled = reconstruction(seed, mask, method='erosion')
	return filled

# Dilate an input image
def dilate(img, KERNEL=np.ones((5,5), np.uint8)):
	dilation = cv2.dilate(img,KERNEL)
	return dilation

# Create an alpha channel for an image
def alpha(r, g, b, OPACITY=50):
	alpha = np.ones(b.shape, dtype=b.dtype) * OPACITY
	alpha[np.where((r[:,:] == 0) & (g[:,:] == 0) & (b[:,:] == 0))] = 0
	return alpha

# Merge r, g, b, and - if present - alpha channels into a 3D or 4D image
def merge(r, g, b, a=None):
	if a.all() == None: 
		img = cv2.merge((r, g, b))
	else: 
		img = cv2.merge((r, g, b, a))
	return img

# Overlay a prediction on a base image
def overlay(img, base, alpha):
	overlay = cv2.addWeighted(img, alpha, base, 1-alpha, 0)
	return overlay

# Return an output .png file
def output_png(img, WIDTH=2, HEIGHT=2, NAME='out.png'):
	fig = plt.figure(frameon = False)
	fig.set_size_inches(WIDTH, HEIGHT)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(img)
	fig.savefig(NAME)
	plt.close()

def img_processing(img, name):
	# Pre-process image to get 3 channel outputs
	r, g, b = pre_process(img)

	# Label and extract properties
	r_labels, r_num = label(r)
	r_props = properties(r_labels)
	g_labels, g_num = label(g)
	g_props = properties(g_labels)
	b_labels, b_num = label(b)
	b_props = properties(b_labels)

	# Smoothen and rectify predictions by deleting noise, filling holes, padding image
	r = dilate(fill_holes(del_noise(r, r_labels, r_num, r_props)))
	g = dilate(fill_holes(del_noise(g, g_labels, g_num, g_props)))
	b = dilate(fill_holes(del_noise(b, b_labels, b_num, b_props)))

	# Output prediction PNG
	output_png(merge(r, g, b, alpha(r, g, b)), NAME=name)

# Call main
if __name__ == "__main__":
	main()