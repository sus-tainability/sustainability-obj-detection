# Sustainability Object Detection

This repository holds our backend for serving our Tensorflow2 object detection model. 

## Motivations

While Sustainability relies on community efforts to validate images submitted by the users, we can **reduce the amount of effort** required by the community and **increase the accuracy of data collected**. 

When a photo is taken by a user, the AI can run inference on these images. For the particular image, if the users validation result correspond to the AI's result, it eliminates the need for other users to look at the image again. Hence, we can **reduce the number of user needed to validate per image taken.**

Furthermore, as more data are collected, these images can be used to retrain the AI and improve its performance. Eventually, we foresee a future where **community validation is no longer needed** when the trained model is robust enough and its error rate is at an acceptable level.

## Technologies

| Tech | Version |
|-:|:-|
| TensorFlow | 2.12.0
| Flask | 2.2.2 

## Usage

The server only exposes a single POST request on {hostname}/api/upload

### Using Curl

    curl -X POST -F file=@"<path to img>" {hostname}/predict

## Setup (Local)

This setup was only tested on `python 3.10.8`

1. Install python dependencies
```sh
pip install -r requirements.txt
```

2. Install TensorFlow's object detection api

Follow instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)

3. (Optional) Setup environment variables

| ENV | Example |
| -: | :- |
| PIPELINE_CONFIG_PATH | configs/ssd_efficientdet_d2_768x768_coco17_tpu-8.config
| CHKPT_DIR | trained_models/efficientdet_d2_coco17_tpu-32/checkpoint
| LABEL_MAP_PATH | label_maps/labels.pbtxt
| DETECT_THRESHOLD | 0.7 

4. Run server
```sh
flask run
```

