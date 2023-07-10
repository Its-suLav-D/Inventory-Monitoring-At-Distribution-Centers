import tensorflow as tf
import os
import json
from PIL import Image
import numpy as np


def model_fn(model_dir):
    model = tf.keras.models.load_model(os.path.join(model_dir, '1'))
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        img_dict = json.loads(request_body)
        img_data = img_dict['url']
        img_array = np.array(img_data)
        return img_array
    else:
        raise ValueError("This model only supports application/json input")



def predict_fn(input_object, model):
    input_object = np.expand_dims(input_object, axis=0)
    prediction = model.predict(input_object)
    return prediction


def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps(prediction.tolist()), response_content_type
    raise ValueError("This model only supports application/json output")
