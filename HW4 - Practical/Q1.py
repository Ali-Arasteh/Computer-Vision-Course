import cv2

template = cv2.imread('images/template.jpg')
image = cv2.imread('images/image.jpg')

# akaze
akaze = cv2.AKAZE_create()
keyPointsTemplate, descTemplate = akaze.detectAndCompute(template, None)
keyPointsImage, descImage = akaze.detectAndCompute(image, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matchedPoints = sorted(bf.match(descTemplate, descImage), key=lambda x: x.distance)
matchedImage = cv2.drawMatches(template, keyPointsTemplate, image, keyPointsImage, matchedPoints[:50], None, flags=2)
cv2.imshow('akaze matched points', matchedImage)
cv2.imwrite('images/AKAZE.jpg', matchedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# orb
orb = cv2.ORB_create()
keyPointsTemplate, descTemplate = orb.detectAndCompute(template, None)
keyPointsImage, descImage = orb.detectAndCompute(image, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matchedPoints = sorted(bf.match(descTemplate, descImage), key=lambda x: x.distance)
matchedImage = cv2.drawMatches(template, keyPointsTemplate, image, keyPointsImage, matchedPoints[:50], None, flags=2)
cv2.imshow("orb matched points", matchedImage)
cv2.imwrite('images/ORB.jpg', matchedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
