import glob
import os

import cv2 as cv
from PIL import Image


def treat_images(input_folder, output_folder="app/treated_images"):
    files = glob.glob(f"{input_folder}/*")

    for file in files:
        image = cv.imread(file)

        gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        _, treated_image = cv.threshold(
            gray_image, 127, 255, cv.THRESH_TRUNC or cv.THRESH_OTSU
        )
        file_name = os.path.basename(file)
        cv.imwrite(f"{output_folder}/{file_name}", treated_image)

    files = glob.glob(f"{output_folder}/*")
    for file in files:
        image = Image.open(file)
        image = image.convert("L")
        image2 = Image.new("L", image.size, 255)

        for x in range(image.size[1]):
            for y in range(image.size[0]):
                pixel_color = image.getpixel((y, x))
                if pixel_color < 115:
                    image2.putpixel((y, x), 0)

        file_name = os.path.basename(file)

        image2.save(f"{output_folder}/{file_name}")


if __name__ == "__main__":
    treat_images("app/db_captcha")
