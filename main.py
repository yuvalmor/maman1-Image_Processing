import cv2
import numpy as np
import sys

LEVEL_VALUES = [0, 36, 72, 108, 144, 180, 216, 255]
VALUES_GAP = 36


def get_image():
    image_name = sys.argv[1]
    return cv2.imread(image_name)


def get_closest_level_value(pixel_value):
    return LEVEL_VALUES[round(pixel_value / VALUES_GAP)]


def display_images(original_image, processed_image):
    cv2.imshow('original_image', original_image)
    cv2.imshow('processed_image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def color_error_diffusion(image):
    height = image.shape[0]
    width = image.shape[1]
    error = np.zeros((height + 1, width + 1))
    processed_image = np.zeros((height, width))
    for x in range(height):
        for y in range(width):
            processed_image[x,y] = get_closest_level_value(image[x,y])
            difference = (image[x,y] + error[x,y]) - processed_image[x,y]
            error[x,y+1] += int(difference*(3/8))
            error[x+1,y] += int(difference*(3/8))
            error[x+1,y+1] += int(difference*(1/4))
    return processed_image


if __name__ == '__main__':
    image = get_image()
    image_b, image_g, image_r = cv2.split(image)
    processed_images = []
    processed_image_b = color_error_diffusion(image_b)
    processed_image_g = color_error_diffusion(image_g)
    processed_image_r = color_error_diffusion(image_r)
    merged_image = cv2.merge((processed_image_b,processed_image_g,processed_image_r))
    display_images(image, merged_image)
    filename = 'savedImage.jpg'
    cv2.imwrite(filename, merged_image)