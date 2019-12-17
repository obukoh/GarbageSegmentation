from PIL import Image, ImageOps

im = Image.open('5441.png')

im_flip = ImageOps.flip(im)
im_flip.save('5441a.png', quality=95)
