

from PIL import Image
import os



def resize_images(input_directory="./input", output_directory="./output", neu_size=(300,300)):
	directory_list = os.listdir(input_directory)
	
	for directory_name in directory_list:
		image_name_list = os.listdir("{}/{}".format(input_directory, directory_name))
		
		for image_name in image_name_list:
			im = Image.open("{}/{}/{}".format(input_directory, directory_name, image_name))
			neu_im = im.resize(neu_size)
			neu_im.save("{}/{}/{}".format(output_directory, directory_name, image_name))
	

if __name__ == '__main__':
	resize_images()


