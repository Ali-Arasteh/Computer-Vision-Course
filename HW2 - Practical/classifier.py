import numpy
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from skimage import data, color, transform, feature
import pickle

faces = fetch_lfw_people()
positive_patches = faces.images
imgs_to_use = ['camera', 'text', 'coins', 'moon', 'page', 'clock', 'immunohistochemistry', 'chelsea', 'coffee', 'hubble_deep_field']
images = [color.rgb2gray(getattr(data, name)()) for name in imgs_to_use]


def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * numpy.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size, max_patches=N, random_state=0)
    patches = extractor.transform(img[numpy.newaxis])
    if scale != 1:
        patches = numpy.array([transform.resize(patch, patch_size) for patch in patches])
    return patches


negative_patches = numpy.vstack([extract_patches(im, 1000, scale) for im in images for scale in [0.5, 1.0, 2.0]])
positive_length = feature.hog(positive_patches[0, :, :]).size
positive_feature = numpy.zeros((positive_patches.shape[0], positive_length))
negative_length = feature.hog(negative_patches[0, :, :]).size
negative_feature = numpy.zeros((negative_patches.shape[0], negative_length))
for i in range(0, positive_patches.shape[0]):
    positive_feature[i, :] = feature.hog(positive_patches[i, :, :])
for i in range(0, negative_patches.shape[0]):
    negative_feature[i, :] = feature.hog(negative_patches[i, :, :])
features = numpy.concatenate((positive_feature, negative_feature))
label = numpy.concatenate((numpy.ones(positive_feature.shape[0]), numpy.zeros(negative_feature.shape[0])))
X_train, X_test, Y_train, Y_test = train_test_split(features, label, train_size=0.8)
classifier = SVC()
classifier.fit(X_train, Y_train)
with open("classifier.file", "wb") as f:
    pickle.dump(classifier, f, pickle.HIGHEST_PROTOCOL)
Y_predicted = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_predicted)
print(accuracy)
#classifier = SVC()
#grid_search = GridSearchCV(estimator=classifier, param_grid=[{'C': [1, 2, 5, 10, 25], 'kernel': ['linear', 'poly', 'rbf']}], scoring='accuracy', cv=10, n_jobs=-1)
#grid_search = grid_search.fit(X_train, Y_train)
#print(grid_search)
#based on GridSearchCV