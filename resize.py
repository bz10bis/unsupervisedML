from PIL import Image
import os

path = "./Hands/"
dirs = os.listdir(path)

i = 0
for item in dirs:
    i = i + 1
    if os.path.isfile(path + item):
        im = Image.open(path + item).convert('LA')
        imResize = im.resize((32, 32), Image.ANTIALIAS)
        imResize.save("./Hands/normalized_32/" + item, 'png')
