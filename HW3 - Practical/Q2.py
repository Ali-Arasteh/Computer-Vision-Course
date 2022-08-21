import numpy
from sklearn.feature_extraction.image import PatchExtractor
from matplotlib import pyplot
from skimage import data, color, transform
from skimage.feature import local_binary_pattern


def extract_patches(img, N, scale=1.0, patch_size=(250, 250)):
    extracted_patch_size = tuple((scale * numpy.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size, max_patches=N, random_state=0)
    patches = extractor.transform(img[numpy.newaxis])
    if scale != 1:
        patches = numpy.array([transform.resize(patch, patch_size) for patch in patches])
    return patches


def lbp_histogram(img, n_points, radius, method):
    lbp = local_binary_pattern(img, n_points, radius, method)
    n_bins = int(lbp.max() + 1)
    hist, _ = numpy.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    return hist


def kullback_score(lbp_in, lbp_train):
    lbp_in = numpy.asarray(lbp_in)
    lbp_train = numpy.asarray(lbp_train)
    f = numpy.logical_and(lbp_train != 0, lbp_in != 0)
    return numpy.sum(lbp_in[f] * numpy.log2(lbp_in[f] / lbp_train[f]))


imgs_to_use = ['brick']
images = [color.rgb2gray(getattr(data, name)()) for name in imgs_to_use]
brickPatches = numpy.vstack([extract_patches(im, 6, scale) for im in images for scale in [1.0]])
fig, ax = pyplot.subplots(2, 3)
for i, axi in enumerate(ax.flat):
    axi.imshow(brickPatches[i], cmap='gray')
    axi.axis('off')
pyplot.show()
imgs_to_use = ['grass']
images = [color.rgb2gray(getattr(data, name)()) for name in imgs_to_use]
grassPatches = numpy.vstack([extract_patches(im, 6, scale) for im in images for scale in [1.0]])
fig, ax = pyplot.subplots(2, 3)
for i, axi in enumerate(ax.flat):
    axi.imshow(grassPatches[i], cmap='gray')
    axi.axis('off')
pyplot.show()
imgs_to_use = ['gravel']
images = [color.rgb2gray(getattr(data, name)()) for name in imgs_to_use]
gravelPatches = numpy.vstack([extract_patches(im, 6, scale) for im in images for scale in [1.0]])
fig, ax = pyplot.subplots(2, 3)
for i, axi in enumerate(ax.flat):
    axi.imshow(gravelPatches[i], cmap='gray')
    axi.axis('off')
pyplot.show()
brick = data.brick()
grass = data.grass()
gravel = data.gravel()
brickLBP = lbp_histogram(brick, 16, 2, 'uniform')
gravelLBP = lbp_histogram(gravel, 16, 2, 'uniform')
grassLBP = lbp_histogram(grass, 16, 2, 'uniform')
label = numpy.zeros(3)
label[0] = 1
label[1] = 2
label[2] = 3
test = numpy.vstack((brickLBP, grassLBP, gravelLBP))
label_test = numpy.zeros(3)
score = numpy.zeros(3)
for i in range(0, 3):
    score[0] = kullback_score(test[i, :], brickLBP)
    score[1] = kullback_score(test[i, :], grassLBP)
    score[2] = kullback_score(test[i, :], gravelLBP)
    label_test[i] = numpy.argmin(score) + 1
if numpy.array_equal(label_test, label):
    print('correct')
