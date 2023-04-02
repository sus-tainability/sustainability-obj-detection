import os
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

def load_model(model_dir, pipeline_config):
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(
        model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

    return detection_model

def load_label_map(label_map_path):
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return category_index