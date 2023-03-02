import os
import cv2
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from util.preprocess import align_for_lm
from shutil import move

mean_face = np.loadtxt('util/test_mean_face.txt')
mean_face = mean_face.reshape([68, 2])

def save_label(labels, save_path):
    np.savetxt(save_path, labels)

def draw_landmarks(img, landmark, save_name):
    landmark = landmark
    lm_img = np.zeros([img.shape[0], img.shape[1], 3])
    lm_img[:] = img.astype(np.float32)
    landmark = np.round(landmark).astype(np.int32)

    for i in range(len(landmark)):
        for j in range(-1, 1):
            for k in range(-1, 1):
                if img.shape[0] - 1 - landmark[i, 1]+j > 0 and \
                        img.shape[0] - 1 - landmark[i, 1]+j < img.shape[0] and \
                        landmark[i, 0]+k > 0 and \
                        landmark[i, 0]+k < img.shape[1]:
                    lm_img[img.shape[0] - 1 - landmark[i, 1]+j, landmark[i, 0]+k,
                           :] = np.array([0, 0, 255])
    lm_img = lm_img.astype(np.uint8)

    cv2.imwrite(save_name, lm_img)


def load_data(img_name, txt_name):
    return cv2.imread(img_name), np.loadtxt(txt_name)

# create tensorflow graph for landmark detector
def load_lm_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='net')
        img_224 = graph.get_tensor_by_name('net/input_imgs:0')
        output_lm = graph.get_tensor_by_name('net/lm:0')
        lm_sess = tf.Session(graph=graph)

    return lm_sess,img_224,output_lm

# landmark detection
def detect_68p(img_path,sess,input_op,output_op):
    print('detecting landmarks......')
    names = [i for i in sorted(os.listdir(
        img_path)) if 'jpg' in i or 'png' in i or 'jpeg' in i or 'PNG' in i]
    vis_path = os.path.join(img_path, 'vis')
    remove_path = os.path.join(img_path, 'remove')
    save_path = os.path.join(img_path, 'landmarks')
    if not os.path.isdir(vis_path):
        os.makedirs(vis_path)
    if not os.path.isdir(remove_path):
        os.makedirs(remove_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i in range(0, len(names)):
        name = names[i]
        print('%05d' % (i), ' ', name)
        full_image_name = os.path.join(img_path, name)
        txt_name = '.'.join(name.split('.')[:-1]) + '.txt'
        full_txt_name = os.path.join(img_path, 'detections', txt_name) # 5 facial landmark path for each image

        # if an image does not have detected 5 facial landmarks, remove it from the training list
        if not os.path.isfile(full_txt_name):
            move(full_image_name, os.path.join(remove_path, name))
            continue 

        # load data
        img, five_points = load_data(full_image_name, full_txt_name)
        input_img, scale, bbox = align_for_lm(img, five_points) # align for 68 landmark detection 

        # if the alignment fails, remove corresponding image from the training list
        if scale == 0:
            move(full_txt_name, os.path.join(
                remove_path, txt_name))
            move(full_image_name, os.path.join(remove_path, name))
            continue

        # detect landmarks
        input_img = np.reshape(
            input_img, [1, 224, 224, 3]).astype(np.float32)
        landmark = sess.run(
            output_op, feed_dict={input_op: input_img})

        # transform back to original image coordinate
        landmark = landmark.reshape([68, 2]) + mean_face
        landmark[:, 1] = 223 - landmark[:, 1]
        landmark = landmark / scale
        landmark[:, 0] = landmark[:, 0] + bbox[0]
        landmark[:, 1] = landmark[:, 1] + bbox[1]
        landmark[:, 1] = img.shape[0] - 1 - landmark[:, 1]

        if i % 100 == 0:
            draw_landmarks(img, landmark, os.path.join(vis_path, name))
        save_label(landmark, os.path.join(save_path, txt_name))
