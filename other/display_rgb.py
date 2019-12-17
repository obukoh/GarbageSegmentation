import cv2
import numpy as np


np.set_printoptions(threshold=np.nan)

def main():

    img = cv2.imread('after/2.png')
    assert img is not None, 'cannot open file as img.'
    height, width, ch = img.shape
    #for a in img[range(0, height), range(0, width)]:
        #print(a)
    for i in range(height):
        for j in range(width):
            print(img[i, j, :], end="")


if __name__ == "__main__":
    main()
