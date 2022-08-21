import cv2

faceClassifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeClassifier = cv2.CascadeClassifier('haarcascade_eye.xml')
smileClassifier = cv2.CascadeClassifier('haarcascade_smile.xml')

# image 1
image = cv2.imread('TestImages/img1.jpg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceClassifier.detectMultiScale(grayImage, 1.2, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    eyes = eyeClassifier.detectMultiScale(grayImage, 1.2, 10)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
    smiles = smileClassifier.detectMultiScale(grayImage, 1.2, 100)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(image, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)
cv2.imshow('image', image)
cv2.imwrite('TestImages/detected img1.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# image 2
image = cv2.imread('TestImages/img2.jpg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceClassifier.detectMultiScale(grayImage, 1.2, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    eyes = eyeClassifier.detectMultiScale(grayImage, 1.2, 10)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
    smiles = smileClassifier.detectMultiScale(grayImage, 1.25, 150)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(image, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)
cv2.imshow('image', image)
cv2.imwrite('TestImages/detected img2.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# image 3
image = cv2.imread('TestImages/img3.jpg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceClassifier.detectMultiScale(grayImage, 1.2, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    eyes = eyeClassifier.detectMultiScale(grayImage, 1.2, 10)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
    smiles = smileClassifier.detectMultiScale(grayImage, 1.2, 100)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(image, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)
cv2.imshow('image', image)
cv2.imwrite('TestImages/detected img3.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# image 4
image = cv2.imread('TestImages/img4.jpg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceClassifier.detectMultiScale(grayImage, 1.2, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    eyes = eyeClassifier.detectMultiScale(grayImage, 1.25, 15)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
    smiles = smileClassifier.detectMultiScale(grayImage, 1.25, 200)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(image, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)
cv2.imshow('image', image)
cv2.imwrite('TestImages/detected img4.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# image 5
image = cv2.imread('TestImages/img5.png')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceClassifier.detectMultiScale(grayImage, 1.2, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    eyes = eyeClassifier.detectMultiScale(grayImage, 1.2, 10)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
    smiles = smileClassifier.detectMultiScale(grayImage, 1.2, 100)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(image, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)
cv2.imshow('image', image)
cv2.imwrite('TestImages/detected img5.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
