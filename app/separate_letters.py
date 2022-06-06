import glob
import os
import cv2 as cv

files = glob.glob("app/treated_images/*")
for file in files:
    image = cv.imread(file)
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    _, new_image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(new_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    area_letters = []

    for contour in contours:
        (x, y, width, height) = cv.boundingRect(contour)
        area = cv.contourArea(contour)
        if area > 115:
            area_letters.append((x, y, width, height))
    if len(area_letters) != 5:
        continue

    final_image = cv.merge([image] * 3)

    for i, rectangle in enumerate(area_letters, 1):
        x, y, width, height = rectangle
        letter_image = image[y : y + height, x : x + width]

        file_name = os.path.basename(file).replace(".png", f"letter{i}.png")

        cv.imwrite(f"app/letters/{file_name}", letter_image)

        cv.rectangle(
            final_image, (x - 2, y - 2), (x + width + 2, y + height + 2), (0, 255, 0), 1
        )

    file_name = os.path.basename(file)
    cv.imwrite(f"app/identified_images/{file_name}", final_image)
