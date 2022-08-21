import cv2
import numpy


image = cv2.imread('image/2.jpg', cv2.IMREAD_GRAYSCALE)
image = image.astype(dtype=numpy.float)
image = cv2.resize(image, (400, 300), interpolation=cv2.INTER_AREA)
extendedImage = numpy.zeros((image.shape[0] + 2, image.shape[1] + 2))
extendedImage[1:-1, 1:-1] = image
lowPassImage = numpy.zeros(image.shape)
xEdging = numpy.zeros(image.shape)
yEdging = numpy.zeros(image.shape)
for i in range(1, image.shape[0] + 1):
    for j in range(1, image.shape[1] + 1):
        lowPassImage[i - 1, j - 1] = 1 / 9 * (extendedImage[i - 1, j - 1] + extendedImage[i - 1, j] + extendedImage[i - 1, j - 1] + extendedImage[i, j - 1] + extendedImage[i, j] + extendedImage[i, j + 1] + extendedImage[i + 1, j - 1] + extendedImage[i + 1, j] + extendedImage[i + 1, j + 1])
        xEdging[i - 1, j - 1] = extendedImage[i + 1, j] - extendedImage[i, j]
        yEdging[i - 1, j - 1] = extendedImage[i, j + 1] - extendedImage[i, j]
highPassImage = cv2.subtract(image, lowPassImage)
lowPassImage = lowPassImage.astype(dtype=numpy.uint8)
xEdging = numpy.absolute(xEdging)
xEdging = xEdging.astype(dtype=numpy.uint8)
yEdging = numpy.absolute(yEdging)
yEdging = yEdging.astype(dtype=numpy.uint8)
modifiedHighPassImage = highPassImage + numpy.absolute(numpy.amin(highPassImage))
modifiedHighPassImage = modifiedHighPassImage * 255 / numpy.amax(modifiedHighPassImage)
modifiedHighPassImage = modifiedHighPassImage.astype(dtype=numpy.uint8)
cv2.imshow('Low pass image', lowPassImage)
cv2.imwrite('image/Low pass image.jpg', lowPassImage)
cv2.imshow('Edging along the x axis', xEdging)
cv2.imwrite('image/Edging along the x axis.jpg', xEdging)
cv2.imshow('Edging along the y axis', yEdging)
cv2.imwrite('image/Edging along the y axis.jpg', yEdging)
cv2.imshow('Modified high pass image', modifiedHighPassImage)
cv2.imwrite('image/Modified high pass image.jpg', modifiedHighPassImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
