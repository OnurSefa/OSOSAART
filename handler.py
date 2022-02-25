import PIL
from PIL import Image
import os
import numpy as np


def resize_images(input_directory="./input", output_directory="./output", neu_size=(300,300)):
	directory_list = os.listdir(input_directory)
	
	for directory_name in directory_list:
		image_name_list = os.listdir("{}/{}".format(input_directory, directory_name))
		
		for image_name in image_name_list:
			try:
				im = Image.open("{}/{}/{}".format(input_directory, directory_name, image_name))
				neu_im = im.resize(neu_size)
				neu_im.save("{}/{}/{}".format(output_directory, directory_name, image_name))
			except OSError:
				print(image_name)


def image_to_numpy(input_directory="./output", output_directory="./numpy_images", label_directory="./labels"):
	directory_list = os.listdir(input_directory)
	for label, directory_name in enumerate(directory_list):
		image_name_list = os.listdir("{}/{}".format(input_directory, directory_name))
		
		for image_name in image_name_list:
			try:
				im = Image.open("{}/{}/{}".format(input_directory, directory_name, image_name))
				np_value = np.array(im)
				# np_result = np.append(np_value, label)
				with open("{}/{}".format(label_directory, image_name.split(".")[0] + ".npy"), 'wb') as output_file:
					np.save(output_file, np.array(label))
				with open("{}/{}".format(output_directory, image_name.split(".")[0] + ".npy"), 'wb') as output_file:
					np.save(output_file, np_value)
			except PIL.UnidentifiedImageError:
				print(image_name)


def take_data(input_directory_1="./numpy_images", input_directory_2="./labels", limit=10000):
	directory_list = os.listdir(input_directory_1)
	data = []
	labels = []
	for i, image_name in enumerate(directory_list):
		try:
			datum = np.load("{}/{}".format(input_directory_1, image_name))
			datum = datum.reshape(3, 300, 300)
			label = np.load("{}/{}".format(input_directory_2, image_name))
			data.append(datum)
			labels.append(label)
		except ValueError:
			# print('valueError')
			pass
		if i % 100 == 0:
			print(i)
		if i == limit:
			break
	return np.array(data), np.array(labels)


if __name__ == '__main__':
	# resize_images('./wiki_images', './resized_wiki_images')
	# image_to_numpy('./resized_wiki_images')
	x, y = take_data()
	# print("a")

