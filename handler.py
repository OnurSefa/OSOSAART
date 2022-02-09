

from PIL import Image
import os
import numpy as np


def resize_images(input_directory="./input", output_directory="./output", neu_size=(300,300)):
	directory_list = os.listdir(input_directory)
	
	for directory_name in directory_list:
		image_name_list = os.listdir("{}/{}".format(input_directory, directory_name))
		
		for image_name in image_name_list:
			im = Image.open("{}/{}/{}".format(input_directory, directory_name, image_name))
			neu_im = im.resize(neu_size)
			neu_im.save("{}/{}/{}".format(output_directory, directory_name, image_name))


styles = ["abstract", "animal-painting", "cityscape", "figurative", "flower-painting"]

def image_to_numpy(input_directory="./output", output_directory="./numpy_images", label_directory="./labels"):
	directory_list = os.listdir(input_directory)
	for directory_name in directory_list:
		image_name_list = os.listdir("{}/{}".format(input_directory, directory_name))
		
		for image_name in image_name_list:
			im = Image.open("{}/{}/{}".format(input_directory, directory_name, image_name))
			np_value = np.array(im)
			label = styles.index(directory_name)
			# np_result = np.append(np_value, label)
			with open("{}/{}".format(label_directory, image_name.split(".")[0] + ".npy"), 'wb') as output_file:
				np.save(output_file, np.array(label))
			with open("{}/{}".format(output_directory, image_name.split(".")[0] + ".npy"), 'wb') as output_file:
				np.save(output_file, np_value)

def take_data(input_directory_1="./numpy_images", input_directory_2="./labels"):
	directory_list = os.listdir(input_directory_1)
	data = []
	labels = []
	for image_name in directory_list:
		datum = np.load("{}/{}".format(input_directory_1, image_name))
		datum = datum.reshape(3, 300, 300)
		label = np.load("{}/{}".format(input_directory_2, image_name))
		data.append(datum)
		labels.append(label)
	return np.array(data), np.array(labels)

    

if __name__ == '__main__':
	x, y = take_data()
	# print("a")

