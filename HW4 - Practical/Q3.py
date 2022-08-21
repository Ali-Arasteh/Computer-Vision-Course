import cv2
import numpy
from copy import deepcopy


def onMouse(event, x, y, flags, params):
    global image
    global temp
    global counter
    global points
    global multiSelect
    if event == cv2.EVENT_LBUTTONDOWN:
        points[counter] = x, y
        temp = cv2.circle(temp, (points[counter][0], points[counter][1]), 2, (0, 0, 255), -1)
        counter = counter + 1
        cv2.imshow('image', temp)
        if counter == 4:
            cv2.destroyAllWindows()
            panel = cv2.imread('images/2-3.jpg')
            srcPoints = numpy.float32(points)
            dstPoints = numpy.float32([[0, 0], [0, panel.shape[0]], [panel.shape[1], panel.shape[0]], [panel.shape[1], 0]])
            PT = cv2.getPerspectiveTransform(dstPoints, srcPoints)
            warpPanel = cv2.warpPerspective(panel, PT, (image.shape[1], image.shape[0]))
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if warpPanel[i, j, 0] != 0:
                        image[i, j, :] = warpPanel[i, j, :]
            cv2.imshow('image', image)
            cv2.imwrite('images/2-1 modified.jpg', image)
            if multiSelect:
                cv2.setMouseCallback('image', onMouse)
                counter = 0
            temp = deepcopy(image)


image = cv2.imread('images/2-1.jpg')
image = cv2.resize(image, (int(image.shape[1] * 0.4), int(image.shape[0] * 0.4)), interpolation=cv2.INTER_AREA)
temp = deepcopy(image)
cv2.imshow('image', temp)
cv2.setMouseCallback('image', onMouse)
points = numpy.zeros((4, 2), numpy.int)
counter = 0
multiSelect = True
cv2.waitKey(0)
cv2.destroyAllWindows()
