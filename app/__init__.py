
import glob
import os

import cv2
from PIL import Image


def treat_images(source_folder, destination_folder='adjusted'):
    files = glob.glob(f'{source_folder}/*')

    for file in files:
        image = cv2.imread(file)

        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, treated_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TRUNC or cv2.THRESH_OTSU)
        file_name = os.path.basename(file)
        cv2.imwrite(f'{destination_folder}/{file_name}', treated_image)

    files = glob.glob(f"{destination_folder}/*")
    for file in files:
        image = Image.open(file)
        image = image.convert("L")
        image2 = Image.new('L', image.size, 255) 

        for x in range(image.size[1]):
            for y in range(image.size[0]):
                pixel_color = image.getpixel((y, x))
                if pixel_color < 115:
                    image2.putpixel((y, x), 0)

        file_name = os.path.basename(file)

        image2.save(f'{destination_folder}/{file_name}')



if __name__ == "__main__":
    treat_images("db_captcha")
