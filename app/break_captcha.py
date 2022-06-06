import os
import pickle

import cv2 as cv
import numpy as np
from imutils import paths
from keras.models import load_model

from helpers import resize_to_fit
from treat_images import treat_images


def break_captcha():
    with open("app/labels_model.dat", "rb") as translator_file:
        lb = pickle.load(translator_file)

    model = load_model("app/trained_model.hdf5")

    treat_images(input_folder="app/solve", output_folder="app/solve")

    files = list(paths.list_images("app/solve"))

    for file in files:
        image = cv.imread(file)
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # current black and white image
        _, new_image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV)

        # find the contours of each letter
        contours, _ = cv.findContours(
            new_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        area_letters = []

        # filter contours that are actually labeled
        for contour in contours:
            (x, y, width, height) = cv.boundingRect(contour)
            area = cv.contourArea(contour)
            if area > 115:
                area_letters.append((x, y, width, height))

        area_letters = sorted(area_letters, key=lambda x: x[0])
        #  drawing the contours and separating the letters into individual files
        final_image = cv.merge([image] * 3)
        output = []

        for i, rectangle in enumerate(area_letters, 1):
            x, y, width, height = rectangle
            letter_image = image[y : y + height, x : x + width]

            letter_image = resize_to_fit(letter_image, 20, 20)

            #  (0 - 255 , 0 - 255) current image dimension
            # (1, 0 - 255, 0 - 255,1) dimension needed for keras to work
            # need to add 2 more dimensions in the image pos[0] and pos[2]

            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            answer_letter = model.predict(letter_image)
            answer_letter = lb.inverse_transform(answer_letter)[0]

            output.append(answer_letter)

            # drawing the predicted letter in the final image
            cv.rectangle(
                final_image,
                (x - 2, y - 2),
                (x + width + 2, y + height + 2),
                (0, 255, 0),
                1,
            )
            file_name = os.path.basename(file)
            cv.imwrite(f"app/solve/{file_name}", final_image)

        output_text = "".join(output)
        print(output_text)
        return output_text


if __name__ == "__main__":
    break_captcha()
