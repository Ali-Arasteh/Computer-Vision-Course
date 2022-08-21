import cv2
import numpy


# A - Q2
def on_change(value):
    rotationMatrix = cv2.getRotationMatrix2D((fixedImage.shape[1] / 2, fixedImage.shape[0] / 2), value, 0.85);
    rotatedImage = cv2.warpAffine(extendedImage, rotationMatrix, (fixedImage.shape[1], fixedImage.shape[0]))
    image = cv2.hconcat([fixedImage, rotatedImage])
    point = rotationMatrix * trackedPoint
    finalImage = cv2.line(image, (20, 36), (int(fixedImage.shape[1] + point[0]), int(point[1])), (0, 0, 255), 2)
    cv2.imshow('Rotating Image', finalImage)


image = cv2.imread('image/space.jpg')
cv2.namedWindow('Rotating Image', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('Degree', 'Rotating Image', 0, 360, on_change)
fixedImage = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.9)))
rotaryImage = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))
imageMargin = numpy.zeros((int((fixedImage.shape[0] - rotaryImage.shape[0]) / 2), fixedImage.shape[1], fixedImage.shape[2]), numpy.uint8)
extendedImage = cv2.vconcat([imageMargin, rotaryImage, imageMargin])
trackedPoint = numpy.matrix([[20], [20 + int((fixedImage.shape[0] - rotaryImage.shape[0]) / 2)], [1]])
on_change(0)
cv2.waitKey(0)
cv2.destroyAllWindows()
