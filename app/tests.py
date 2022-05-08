import cv2
from PIL import Image

# methodos de tratamento para imagem
methods = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC,
    cv2.THRESH_TOZERO, 
    cv2.THRESH_TOZERO_INV
]

# lendo a imagem base e guardando em uma variavel
image = cv2.imread("app/tests_method/base_image.png")

# transformando a imagem em escala de cinza
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# percorrendo a lista com os methods para as imagens 
for i, method in enumerate(methods, 1):
    # tratando as imagens com cada metodo
    _, treated_image = cv2.threshold(gray_image, 127, 255, method or cv2.THRESH_OTSU)
    # salvando a imagem na pasta tests_method
    cv2.imwrite(f'app/tests_method/method_{i}.png', treated_image)


# lendo a imagem
image = Image.open('app/tests_method/method_3.png')

# convertendo mais uma vez por garantia
image = image.convert('L')

# gerando uma imagem em branco no mesmo tamanho da imagem atual
# image2 = Image.new('method', image_size, white) 
image2 = Image.new('L', image.size, 255) 

# percorrendo cada pixel da matrix da imagem x e y gerando uma copia.
for x in range(image.size[1]):
    for y in range(image.size[0]):
        # pegando o pixel atual percorrido
        pixel_color = image.getpixel((y, x))
        # se o tom RGB de cinza < 115 
        if pixel_color < 115:
            # pinto o pixel de preto
            image2.putpixel((y, x), 0)

# salvando a imagem tratada
image2.save('app/tests_method/treated_image.png')