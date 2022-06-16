import numpy as np
from HaarLikeFeature import HaarLikeFeature
from HaarLikeFeature import feature_types
from integralimage import integral_image

def create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height,
                    max_feature_height):
    generated_features = []
    for feature in feature_types:
        feature_start_width = max(min_feature_width, feature_types[feature][0])
        for feature_width in range(feature_start_width, max_feature_width, feature_types[feature][0]):
            feature_start_height = max(min_feature_height, feature_types[feature][0])
            for feature_height in range(feature_start_height, max_feature_height, feature_types[feature][1]):
                for x in range(img_width - feature_width):
                    for y in range(img_height - feature_height):
                        generated_features.append(HaarLikeFeature(feature, y, x, feature_width, feature_height, 0, 1))
                        generated_features.append(HaarLikeFeature(feature, y, x, feature_width, feature_height, 0, -1))
    print("generated ", len(generated_features), " features")
    return generated_features


def learn(faces, nonfaces, num_classifiers = -1, min_feature_height = 1, max_feature_height = -1 , min_feature_width = 1 , max_feature_width = -1):

    # list of training data with their labels and weights
    # for each feature find optimum threshold to minimise error
    # select best classifier
    # re-weight all data
    # repeat
    faces = integral_image(faces)
    nonfaces = integral_image(nonfaces)
    num_faces = len(faces)
    num_nonfaces = len(nonfaces)
    num_imgs = num_faces + num_nonfaces
    img_height , img_width = faces[0].shape

    #default max feature height and width is same as that of image
    max_feature_height = img_height if max_feature_height == -1 else max_feature_height
    max_feature_width = img_width if max_feature_width ==-1 else max_feature_width

    # creating array (image , label , weight)
    positive = np.hstack((faces, np.ones(num_faces), (np.ones(num_faces)*(1/(2*num_faces)))))
    negative = np.hstack((nonfaces, np.ones(num_nonfaces)*-1, (np.ones(num_nonfaces)*(1/(2*num_nonfaces)))))
    data = positive + negative

    # create weak classifiers
    classifiers = create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height, max_feature_height)

    # train
    for t in range(num_classifiers): #required classifier
        # normalize the weights
        best_classifiers = []
        data[2] = data[2]/data[2].sum()
        # for each classifier find error
        errors = np.zeros(len(classifiers))
        for c in range(classifiers): #generated classifiers
            classifier = classifier[c]
            error = 0
            for i in data[0]:
                h = classifier.predict(data[0][i])
                error += data[2][i] * abs(h - data[1][i])
            errors[c] = error
        # select best classifier
        best_classifiers + = np.append(best_classifiers, classifiers[np.argmin(errors)])

        # update weights

    return 0
