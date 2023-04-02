import os
import cv2
import numpy as np
import tensorflow as tf
from utils import *
import json

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
 
from flask import Flask, request
app = Flask(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable gpu

temp_dir = 'temp'
pipeline_config = os.environ.get("PIPELINE_CONFIG_PATH")
model_dir = os.environ.get("CHKPT_DIR")
label_map_path = os.environ.get("LABEL_MAP_PATH")

threshold = float(os.environ.get("DETECT_THRESHOLD"))

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
      model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

# load label map
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
     
@app.route('/api/upload', methods=['POST'])
def upload():
    # receive the file
    file = request.files['file']
    filepath = os.path.join(temp_dir, file.filename)
    file.save(filepath) # save to directory
    
    # read image 
    img = cv2.imread(filepath)
    h, w, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)

    # prediction
    image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    results = set()

    classes = (detections['detection_classes'][0].numpy() + 1).astype(int)
    scores = detections['detection_scores'][0].numpy()

    for score, cls in zip(scores, classes):
        if score >= threshold:
            results.add(category_index.get(cls)['name'])

    return json.dumps(list(results))


# Run flask server
if __name__ == '__main__':
    if (not os.path.exists(temp_dir)):
        os.makedirs(temp_dir)

    app.run(debug=True) # set debug true to load reload server auto on changes