import lightgbm as lgb
import pandas as pd
from object_detector import *
from relationship_detector.utils import *

bst = lgb.Booster(model_file=RELATIONSHIP_DETECTION_MODEL_PATH)
targets = ['interacts_with', 'holds', 'on', 'wears', 'inside_of', 'at', 'plays']
labels = dict(zip(range(desc.shape[0]), desc.LabelName))


def get_input(objects):
    df = get_relationship_df(objects, relationship_groups)
    df = add_features(df)
    return df.to_numpy()


def get_predictions(objects):
    if len(objects) == 0:
        return [], [], []

    X = get_input(objects)
    y = bst.predict(X)

    predictions, scores = get_predictions_and_scores(y)
    return X, predictions, scores


def predict_image(image_name):
    objects = get_image_objects(image_name)
    X, predictions, scores = get_predictions(objects)

    relationships = []

    for idx, score in enumerate(scores):
        if score < 0.3:
            continue
        
        X_list = X[idx].tolist()
        X_list[0] = labels[int(X_list[0])]
        X_list[1] = labels[int(X_list[1])]
        relationships.append(X_list + [targets[predictions[idx]], score])

    relationships.sort(key=lambda x: x[-1], reverse=True)
    relationships = relationships[:5]
    visualize_relationships(relationships, image_name, labels, targets)
    