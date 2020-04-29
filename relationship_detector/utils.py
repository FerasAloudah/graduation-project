import pandas as pd
import numpy as np
import cv2
import random

features = ['LabelName1', 'LabelName2', 'XMin1', 'XMax1', 'YMin1', 'YMax1', 'XMin2', 'XMax2', 'YMin2', 'YMax2', 
    'XMin3', 'XMax3', 'YMin3', 'YMax3', 'Area1', 'Area2', 'Area3', 'DistanceTopLeft', 'DistanceTopRight', 'DistanceBottomLeft', 
    'DistanceBottomRight', 'DistanceCenter', 'IoU']


def get_relationship_pairs(objects):
    pairs = []
    for idx, object1 in enumerate(objects[:-1]):
        for object2 in objects[idx+1:]:
            pairs.append([object1, object2])
    return pairs


def get_relationship_rows(pairs, relationship_groups):
    rows = []
    for relationship in pairs:
        row = {}
        if (relationship[0][0], relationship[1][0]) not in relationship_groups:
            relationship = relationship[::-1]
        for idx, object_features in enumerate(relationship):
            row[f'LabelName{idx+1}'] = object_features[0]
            row[f'XMin{idx+1}'] = object_features[1]
            row[f'XMax{idx+1}'] = object_features[2]
            row[f'YMin{idx+1}'] = object_features[3]
            row[f'YMax{idx+1}'] = object_features[4]
        rows.append(row)
    return rows


def get_relationship_df(objects, relationship_groups):
    relationship_pairs = get_relationship_pairs(objects)
    rows = get_relationship_rows(relationship_pairs, relationship_groups)
    return pd.DataFrame.from_dict(rows)


def add_rel_bounding_boxes(df):
    df['XMin3'] = df[['XMin1', 'XMin2']].min(axis=1)
    df['XMax3'] = df[['XMax1', 'XMax2']].max(axis=1)
    df['YMin3'] = df[['YMin1', 'YMin2']].min(axis=1)
    df['YMax3'] = df[['YMax1', 'YMax2']].max(axis=1)
    return df


def add_area(df):
    for i in range(1, 4):
        df[f'Area{i}'] = (df[f'XMax{i}'] - df[f'XMin{i}']) * (df[f'YMax{i}'] - df[f'YMin{i}'])
    return df


def get_box(row, i):
    return {
        'left': row[f'XMin{i}'],
        'top': row[f'YMin{i}'],
        'width': row[f'XMax{i}'] - row[f'XMin{i}'],
        'height': row[f'YMax{i}'] - row[f'YMin{i}']
    }


def intersection_over_union(row):
    box_a = get_box(row, 1)
    box_b = get_box(row, 2)
    # Determine the coordinates of each of the two boxes
    xA = max(box_a['left'], box_b['left'])
    yA = max(box_a['top'], box_b['top'])
    xB = min(box_a['left'] + box_a['width'], box_b['left']+box_b['width'])
    yB = min(box_a['top'] + box_a['height'], box_b['top']+box_b['height'])

    # Calculate the area of the intersection area
    area_of_intersection = (xB - xA + 1) * (yB - yA + 1)

    # Calculate the area of both rectangles
    box_a_area = (box_a['width'] + 1) * (box_a['height'] + 1)
    box_b_area = (box_b['width'] + 1) * (box_b['height'] + 1)
    # Calculate the area of intersection divided by the area of union
    # Area of union = sum both areas less the area of intersection
    iou = area_of_intersection / float(box_a_area + box_b_area - area_of_intersection)

    # Return the score
    return iou


def add_distance(df):
    df['DistanceTopLeft'] = ((df['XMin2'] - df['XMin1']) ** 2 + (df['YMin2'] - df['YMax1']) ** 2) ** 0.5
    df['DistanceTopRight'] = ((df['XMax2'] - df['XMax1']) ** 2 + (df['YMax2'] - df['YMax1']) ** 2) ** 0.5
    df['DistanceBottomLeft'] = ((df['XMin2'] - df['XMin1']) ** 2 + (df['YMin2'] - df['YMin1']) ** 2) ** 0.5
    df['DistanceBottomRight'] = ((df['XMax2'] - df['XMax1']) ** 2 + (df['YMin2'] - df['YMin1']) ** 2) ** 0.5
    df['DistanceCenter'] = (df['DistanceTopLeft'] + df['DistanceTopRight'] + df['DistanceBottomLeft'] + df['DistanceBottomRight']) / 4
    return df


def add_features(df):
    df = add_rel_bounding_boxes(df)
    df = add_area(df)
    df = add_distance(df)
    df['IoU'] = df.apply(intersection_over_union, axis=1)

    df = df[features]

    df.to_csv('test3.csv', index=False)

    return df


def get_predictions_and_scores(y):
    predictions = []
    scores = []
    for idx, prediction in enumerate(y):
        index = np.where(prediction == np.max(prediction))[0][0]
        predictions.append(index)
        scores.append(prediction[index])
    return predictions, scores


def add_bounding_box(image, text, start_point, end_point):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    thickness = 3

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    
    start_point = (start_point[0] + 5, start_point[1] + 35)
    image = cv2.putText(image, text, start_point, font,  
                       font_scale, color, thickness, cv2.LINE_AA)
    return image


def visualize_relationships(relationships, image_name, labels, targets):
    image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2]

    for relation in relationships:
        x1 = (int(width * relation[2]), int(width * relation[3]))
        y1 = (int(height * relation[4]), int(height * relation[5]))

        x2 = (int(width * relation[6]), int(width * relation[7]))
        y2 = (int(height * relation[8]), int(height * relation[9]))

        relationship_start = (min(x1[0], x2[0]), min(y1[0], y2[0]))
        relationship_end = (max(x1[1], x2[1]), max(y1[1], y2[1]))
        image = add_bounding_box(image, f'{relation[0]} {relation[-2]} {relation[1]}', relationship_start, relationship_end)
    
    cv2.imwrite(image_name, image) 