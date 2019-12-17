from PIL import Image, ImageOps
import glob
import os

dir_original = "/home/gonken/okubo_sotsuken/segmentation_test/dataset_unity/newBefore2"
dir_segmented = "/home/gonken/okubo_sotsuken/segmentation_test/dataset_unity/newAfter2"

paths_original = list(glob.glob(dir_original + "/*"))
paths_segmented = list(glob.glob(dir_segmented + "/*"))
i = 5441
for x in paths_original:
    im = Image.open(x)
    im_flip = ImageOps.flip(im)
    im_flip.save('/home/gonken/okubo_sotsuken/segmentation_test/dataset_unity/newBefore3/'+str(i)+'.png', quality=95)
    i += 1
    im.close()

j = 5441
for x in paths_segmented:
    im = Image.open(x)
    im_flip = ImageOps.flip(im)
    im_flip.save('/home/gonken/okubo_sotsuken/segmentation_test/dataset_unity/newAfter3/'+str(j)+'.png', quality=95)
    j += 1
    im.close()

#im = Image.open('data/src/lena.jpg')

#im_flip = ImageOps.flip(im)
#im_flip.save('data/dst/lena_flip.jpg', quality=95)

#im_mirror = ImageOps.mirror(im)
#im_mirror.save('data/dst/lena_mirror.jpg', quality=95)
