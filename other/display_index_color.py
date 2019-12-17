import numpy as np
from PIL import Image
import codecs


image = Image.open("1.png")
image2 = np.asarray(image)
image2.reshape(1,-1)
np.set_printoptions(threshold=np.inf)
b = image2.flatten() # 1次元配列に変形したものをbに代入
print(b,end="")

with codecs.open('result.txt', 'w', 'utf-8') as f:
    f.write(str(image2))

