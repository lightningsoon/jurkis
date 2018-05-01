# coding: utf-8
# 单进程测试用
import numpy as np
import os
import tensorflow as tf
import cv2

if tf.__version__ < '1.5.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util
import core
# Path to frozen detection graph.
PATH_TO_CKPT = os.path.join('/home/momo/catkin_ws/src/jurvis/scripts/Program/Detection/model', 'frozen_inference_graph.pb')
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/momo/catkin_ws/src/jurvis/scripts/Program/Detection/model/', 'mscoco_label_map.pbtxt')
# 类别数量
NUM_CLASSES = 90

# ## 载入标签图
# print(PATH_TO_LABELS)
# print(os.path.isfile(PATH_TO_LABELS),os.path.isfile(PATH_TO_CKPT))
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## 载入模型
od_graph_def = tf.GraphDef()
with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')#和后面有关系


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# detection_graph.as_default()
sess=tf.Session()
tf_get_default_graph=tf.get_default_graph()
tensor_dict = {}
for key in [
    'num_detections', 'detection_boxes', 'detection_scores',
    'detection_classes'
]:
    tensor_name = key + ':0'
    tensor_dict[key] = tf_get_default_graph.get_tensor_by_name(
        tensor_name)
def inference(image):
    # The following processing is only for single image
    image_tensor = tf_get_default_graph.get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image,0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    ret, frame = cap.read()
    core.constant(frame,category_index)
    while True:
        ret, frame = cap.read()
        # Actual detection.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_dict = inference(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Visualization of the results of a detection.
        core.master(
            frame,
            output_dict['num_detections'],
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
        max_boxes_to_draw=3)
        # print(output_dict)
        cv2.imshow('',frame)
        flag=cv2.waitKey(30)
        if flag==27:
            cap.release()
            cv2.destroyAllWindows()
            break
        pass


# 下面是给ros用的
def properity(height,width):
    core.constant4ros(category_index,height,width)
def minor(frame):
    # minor是main的青春版
    '''
    识别
    :param frame:
    :return:
    '''
    # 5 目标检测
    output_dict = inference(frame)
    # Visualization of the results of a detection.
    GraspPoint2D,difference=core.master(
        frame,
        output_dict['num_detections'],
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        max_boxes_to_draw=5)
    # print(output_dict)
    return frame,GraspPoint2D,difference

if __name__ == '__main__':
    main()

'''
[ 0.02503505  0.39374581  0.97495782  1.        ]
[ 0.02503738  0.          0.97495532  0.60621828]
[ 0.39521855  0.55530763  0.8548339   1.        ]


'''
