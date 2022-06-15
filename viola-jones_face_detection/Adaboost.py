from HaarLikeFeature import HaarLikeFeature
from HaarLikeFeature import feature_types


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

def learn(pos, neg, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width):
    return 0
