import cv2
import numpy

# image 1_1
image = cv2.imread('images/1-1.JPG')
image = cv2.resize(image, (int(image.shape[1] * 0.4), int(image.shape[0] * 0.4)), interpolation=cv2.INTER_AREA)
height = 600
width = 1000
srcPoints = numpy.array([[238, 462], [340, 382], [714, 564], [716, 450]])
dstPoints = numpy.array([[50, 50], [300, 50], [50, 550], [300, 550]])
H, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
image = cv2.warpPerspective(image, H, (width, height))
cv2.imshow('image', image)
cv2.imwrite('images/1-1 above view.JPG', image)
cv2.waitKey(0)

# image 1_2
image = cv2.imread('images/1-2.JPG')
image = cv2.resize(image, (int(image.shape[1] * 0.4), int(image.shape[0] * 0.4)), interpolation=cv2.INTER_AREA)
height = 325
width = 650
srcPoints = numpy.array([[432, 315], [575, 486], [244, 336], [274, 545]])
dstPoints = numpy.array([[0, 0], [600, 0], [0, 300], [600, 300]])
H, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
image = cv2.warpPerspective(image, H, (width, height))
cv2.imshow('image', image)
cv2.imwrite('images/1-2 above view.JPG', image)
cv2.waitKey(0)