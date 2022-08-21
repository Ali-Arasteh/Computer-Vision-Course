import cv2
import numpy


image = cv2.imread('image/limbo.png')
kernel = numpy.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], numpy.uint8)
dilatedImage = cv2.dilate(image, kernel, iterations=1)
erodedImage = cv2.erode(image, kernel, iterations=1)
closedImage = cv2.erode(dilatedImage, kernel, iterations=1)
openedImage = cv2.dilate(erodedImage, kernel, iterations=1)
cv2.imshow('Eroded Image', erodedImage)
cv2.imwrite('image/Eroded Image.jpg', erodedImage)
cv2.imshow('Dilated Image', dilatedImage)
cv2.imwrite('image/Dilated Image.jpg', dilatedImage)
cv2.imshow('Closed Image', closedImage)
cv2.imwrite('image/Closed Image.jpg', closedImage)
cv2.imshow('Opened Image', openedImage)
cv2.imwrite('image/Opened Image.jpg', openedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
