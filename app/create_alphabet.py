import string
import os

asc = list(string.ascii_uppercase)

for letter in asc:
    path = f"app/base_letters/{letter}"
    if not os.path.exists(path):
        os.mkdir(path)
