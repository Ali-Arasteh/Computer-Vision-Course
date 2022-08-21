import cv2
import numpy
import pickle

image = cv2.imread('Test_images/test_image1.jpg', 0)
image = image.astype(numpy.uint8)
path = 'Test_images/test_image1'


def HOGDescriptor(image, path):
    hog = cv2.HOGDescriptor()
    features = hog.compute(image)
    with open(path + ".file", "wb") as f:
        pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)


HOGDescriptor(image, path)
