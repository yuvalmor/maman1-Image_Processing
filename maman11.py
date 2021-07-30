import cv2
import numpy as np
import sys

LEVEL_VALUES = [0, 36, 72, 108, 144, 180, 216, 255]
VALUES_GAP = 36


def get_image():
    image_name = sys.argv[1]
    return cv2.imread(image_name)


def color_error_diffusion(image):
    height = image.shape[0]
    width = image.shape[1]
    error = np.zeros((height + 1, width + 1))
    processed_image = np.zeros((height, width), "uint8")
    for x in range(height):
        for y in range(width):
            processed_image[x, y] = get_closest_level_value(image[x, y] + error[x, y])
            difference = (image[x, y] + error[x, y]) - processed_image[x, y]
            error[x, y + 1] += difference * (3 / 8)
            error[x + 1, y] += difference * (3 / 8)
            error[x + 1, y + 1] += difference * (1 / 4)
    return processed_image


def get_closest_level_value(pixel_value):
    return LEVEL_VALUES[round(pixel_value / VALUES_GAP)]


def display_images(original_image, processed_image):
    cv2.imshow('original_image', original_image)
    cv2.imshow('processed_image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = get_image()
    processed_images = []
    for channel in cv2.split(image):
        processed_image = color_error_diffusion(channel)
        processed_images.append(processed_image)
    merged_image = cv2.merge(processed_images)
    display_images(image, merged_image)
    cv2.imwrite('processedImage.png', merged_image)
