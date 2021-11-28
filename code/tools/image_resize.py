import cv2
import os


def resize(img, dim: tuple = (512, 512)):
    resized = cv2.resize(img, dim)
    return resized


def save(img, fn):
    cv2.imwrite(fn, img)


if __name__ == "__main__":
    for i in os.listdir("."):
        if os.path.isfile(i):
            img = cv2.imread(i)
            resized = resize(img, (512, 512))
            fn = i
            save(resized, fn)
