import cv2
import numpy


def drawline(img1, img2, line, point):
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [img1.shape[1], -(line[2] + line[0] * img1.shape[1]) / line[1]])
    color = tuple(numpy.random.randint(0, 255, 3).tolist())
    img1 = cv2.circle(img1, tuple(point), 2, color, -1)
    img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
    return img1, img2


image1 = cv2.imread('images/4-1.jpg')
points78 = cv2.findChessboardCorners(image1, (7, 8))[1]
points65 = cv2.findChessboardCorners(image1, (6, 5))[1]
points1 = numpy.row_stack((points78, points65))
for i in range(points1.shape[0]):
    image1 = cv2.circle(image1, (points1[i][0][0], points1[i][0][1]), 2, (255, 0, 0), -1)
cv2.imshow('4_1', image1)
cv2.imwrite('images/4-1 chessboard corners.jpg', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
image2 = cv2.imread('images/4-2.jpg')
points78 = cv2.findChessboardCorners(image2, (7, 8))[1]
points65 = cv2.findChessboardCorners(image2, (6, 5))[1]
points2 = numpy.row_stack((points78, points65))
for i in range(points2.shape[0]):
    image2 = cv2.circle(image2, (points2[i][0][0], points2[i][0][1]), 2, (255, 0, 0), -1)
cv2.imshow('4_2', image2)
cv2.imwrite('images/4-2 chessboard corners.jpg', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

fundamental = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)[0]

image3 = cv2.imread('images/4-3.jpg')
image4 = cv2.imread('images/4-4.jpg')
point = numpy.array([265, 305]).reshape(-1, 2)
lineCoefficient = cv2.computeCorrespondEpilines(point, 2, fundamental).reshape(3)
image34, image3 = drawline(image4, image3, lineCoefficient, point.reshape(2))
cv2.imshow('4_3', image3)
cv2.imwrite('images/4-3 modified.jpg', image3)
cv2.imshow('4_4', image4)
cv2.imwrite('images/4-4 modified.jpg', image4)
cv2.waitKey(0)
cv2.destroyAllWindows()

# points of image 3
image3 = cv2.imread('images/4-3.jpg')
image4 = cv2.imread('images/4-4.jpg')
for i in range(points1.shape[0]):
    point = points1[i, :, :]
    lineCoefficient = cv2.computeCorrespondEpilines(point, 1, fundamental).reshape(3)
    image3, image4 = drawline(image3, image4, lineCoefficient, point.reshape(2))
cv2.imshow('4_3', image3)
cv2.imwrite('images/4-3 chessboard corners.jpg', image3)
cv2.imshow('4_4', image4)
cv2.imwrite('images/4-4 epipolar lines.jpg', image4)
cv2.waitKey(0)
cv2.destroyAllWindows()

# points of image 4
image3 = cv2.imread('images/4-3.jpg')
image4 = cv2.imread('images/4-4.jpg')
for i in range(points2.shape[0]):
    point = points2[i, :, :]
    lineCoefficient = cv2.computeCorrespondEpilines(point, 2, fundamental).reshape(3)
    image4, image3 = drawline(image4, image3, lineCoefficient, point.reshape(2))
cv2.imshow('4_3', image3)
cv2.imwrite('images/4-3 epipolar lines.jpg', image3)
cv2.imshow('4_4', image4)
cv2.imwrite('images/4-4 chessboard corners.jpg', image4)
cv2.waitKey(0)
cv2.destroyAllWindows()
