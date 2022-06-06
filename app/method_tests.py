import cv2 as cv
from PIL import Image

# image processing methods
methods = [
    cv.THRESH_BINARY,
    cv.THRESH_BINARY_INV,
    cv.THRESH_TRUNC,
    cv.THRESH_TOZERO,
    cv.THRESH_TOZERO_INV,
]

# reading the base image and storing it in a variable
image = cv.imread("app/method_tests/base_image.png")

# converting the image to grayscale
gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

# scrolling through the list with the methods for the images
for i, method in enumerate(methods, 1):
   # manipulating images with each method in the list
    _, treated_image = cv.threshold(gray_image, 127, 255, method or cv.THRESH_OTSU)
    # saving the image in the test_method folder
    cv.imwrite(f"app/method_tests/method_{i}.png", treated_image)


# reading the image
image = Image.open("app/method_tests/method_3.png")

# converting once again for guarantee
image = image.convert("L")

# generating a blank image with the same size as the current image
image2 = Image.new("L", image.size, 255)

# traversing each pixel of the x and y image matrix generating a copy
for x in range(image.size[1]):
    for y in range(image.size[0]):
        # getting the current pixel
        pixel_color = image.getpixel((y, x))
        # if the RGB shade of gray < 115
        if pixel_color < 115:
            # paint the pixel black
            image2.putpixel((y, x), 0)

# saving the treated image
image2.save("app/method_tests/treated_image.png")
