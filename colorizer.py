# CS194-26 : Project 1 
# George Geng

import numpy as np
import skimage as sk
import skimage.transform as transform
import skimage.io as skio
import skimage.filters as filters
import os



all_images = os.listdir('images')


def read_and_split_image(image_name):
	imname = 'images/' + image_name
	im = skio.iread(imname)
	im = sk.img_as_float(im)
	height = int(np.floor(im.shape[0] / 3.0))
	b = im[:height]
	g = im[height: 2*height]
	r = im[2*height: 3*height]
	return [b, g, r]

# name of the input file
imname = 'images/cathedral.jpg'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
height = int(np.floor(im.shape[0] / 3.0))

# extract seperate color images
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# crop borders by 20%, or 10% on each side
def crop(image, crop_percent = 20):
	x_left = int(crop_percent * 0.01 * image.shape[0] * 0.5)
	x_right = image.shape[0] - x_left
	y_left = int(crop_percent * 0.01 * image.shape[1] * 0.5)
	y_right = image.shape[1] - y_left
	return image[x_left:x_right, y_left:y_right]


# align the images
# np.roll, np.sum for ssd
# sk.transform.rescale (for multiscale)

def ssd(image_one, image_two):
	return np.sum(np.square(np.subtract(image_one, image_two))) 

def ncc(image_one, image_two):
	corr = np.multiply(np.divide(image_one,np.linalg.norm(image_one)), np.divide(image_two,np.linalg.norm(image_two)))
	return np.sum(corr)

# align r with b, then g with b, goal to minimize SSD
# only look in middle 2/3rds of the image, so crop by 33% or so
def ssd_align(image_one, image_two, search_window = 15, start_x = 0, start_y = 0, isPyramid = False):
	if not isPyramid: 
		image_one_cropped = crop(image_one, 33)
		image_two_cropped = crop(image_two, 33)
	else:
		image_one_cropped = image_one
		image_two_cropped = image_two

	min_ssd = np.inf
	best_x = 0
	best_y = 0
	x_search_window = search_window # min(search_window, image_one_cropped.shape[0]/2)
	y_search_window = search_window # min(search_window, image_one_cropped.shape[1]/2)
	#print(search_window)

	x_search = range(-x_search_window, x_search_window)
	y_search = range(-y_search_window, y_search_window)
	for x in x_search:
		for y in y_search:
			image_one_shifted = np.roll(image_one_cropped, x, axis = 1)
			image_one_shifted = np.roll(image_one_shifted, y, axis = 0)
			curr_ssd = ssd(image_one_shifted, image_two_cropped)
			if (curr_ssd < min_ssd):
				min_ssd = curr_ssd
				best_x = x
				best_y = y
	best_image = np.roll(image_one, best_x, axis = 1)
	best_image = np.roll(best_image, best_y, axis = 0)
	print(best_x, best_y)
	return [(best_x, best_y), best_image]


# align r with b, then g with b, goal to maximize NCC
# only look in middle 2/3rds of the image, so crop by 33% or so
def ncc_align(image_one, image_two, search_window = 15, start_x = 0, start_y = 0):
	image_one_cropped = crop(image_one, 33)
	image_two_cropped = crop(image_two, 33)
	max_ncc = 0
	best_x = 0
	best_y = 0

	for x in range(-search_window, search_window):
		for y in range(-search_window, search_window): 
			image_one_shifted = np.roll(image_one_cropped, x, axis = 1)
			image_one_shifted = np.roll(image_one_shifted, y, axis = 0)
			curr_ncc = ncc(image_one_shifted, image_two_cropped)
			if (curr_ncc > max_ncc):
				max_ncc = curr_ncc 
				best_x = x
				best_y = y
	best_image = np.roll(image_one, best_x, axis = 1)
	best_image = np.roll(best_image, best_y, axis = 0)
	return [(best_x, best_y), best_image]

# uses pyramiding and one of the above metrics to create
def pyramid_align(image_one, image_two, search_window = 15):
	image_one_cropped = crop(image_one, 33)
	image_two_cropped = crop(image_two, 33)
	best_x = 0
	best_y = 0
	image_one_pyramids = create_pyramids(image_one_cropped,[])
	image_two_pyramids = create_pyramids(image_two_cropped,[])

	for i in range(len(image_one_pyramids)):
		best_x = 2 * best_x
		best_y = 2 * best_y
		curr_pyramid = image_one_pyramids[i]
		curr_pyramid = np.roll(curr_pyramid, best_x, axis = 1)
		curr_pyramid = np.roll(curr_pyramid, best_y, axis = 0)
		align = ssd_align(curr_pyramid, image_two_pyramids[i], start_x = 0, start_y = 0, isPyramid = True)
		best_x += align[0][0]
		best_y += align[0][1]
		print (best_x, best_y)

	best_image = np.roll(image_one, best_x, axis = 1)
	best_image = np.roll(best_image, best_y, axis = 0)
	return best_image

def make_smol_by_half(image): 
	new_x = int(0.5 * image.shape[0])
	new_y = int(0.5 * image.shape[1])
	return transform.resize(image, (new_x, new_y))	

def create_pyramids(image, all_images):
	all_images.insert(0, image)
	if max(image.shape[0], image.shape[1]) < 100:
		return all_images
	else:
		return create_pyramids(make_smol_by_half(image), all_images)

# test = create_pyramids(b, [])
# for t in test:
# 	skio.imshow(t)
# 	skio.show()

# returns a list of all possible images, from coarsest to finest
# def create_all_pyramids(image):
# 	coarsest = np.floor(np.log2(image.shape))
# 	num_levels = min(coarsest[0], coarsest[1])
# 	return pyramids([], image, num_levels)

# # recursively returns a list of scaled images
# def pyramids(all_images, image, num_levels):
# 	all_images.insert(0,image)
# 	if num_levels == 1:
# 		return all_images
# 	else:
# 		return pyramids(all_images, make_smol_by_half(image), num_levels-1)

# code to create the image
# g = crop(g)
# b = crop(b)
# r = crop(r)

ag = pyramid_align(g, b)
print "aligned g"
ar = pyramid_align(r, b)
print "aligned r"
# create a color image
im_out = crop(np.dstack([ar, ag, b]), 20)

# simple_stack = np.dstack([r, g, b])
#smol = make_smol_by_half(im_out)
#skio.imshow(smol)
#skio.show()

# save the image
fname = 'results/self_portrait_pyramid_stack.jpg'
skio.imsave(fname, im_out)

# display the image
skio.imshow(im_out)
skio.show()