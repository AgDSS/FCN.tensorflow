import FCN
import ConfigParser
import tensorflow as tf
import os
import threading


config = ConfigParser.ConfigParser()
config.read("settings.ini")


FCN.FLAGS.batch_size = config.getint("FCN Settings", 'batch_size')
FCN.FLAGS.learning_rate = config.getfloat("FCN Settings", 'learning_rate')
FCN.FLAGS.debug = config.getboolean("FCN Settings", 'debug')
FCN.FLAGS.image_augmentation = config.getboolean("FCN Settings", 'image_augmentation')
FCN.FLAGS.dropout = config.getfloat("FCN Settings", 'dropout')
FCN.FLAGS.mode = config.get("FCN Settings", 'mode')
FCN.MAX_ITERATION = int(config.getfloat("FCN Settings", 'max_iterations')) #Allows for scientific notation, e.g. 1e4
FCN.NUM_CLASSES = config.getint("FCN Settings", 'num_classes')
FCN.IMAGE_WIDTH = config.getint("FCN Settings", 'image_width')
FCN.IMAGE_HEIGHT = config.getint("FCN Settings", 'image_height')

FCN.FLAGS.data_dir = "/dataset"
FCN.FLAGS.logs_dir = "/logs"



#Based on https://stackoverflow.com/questions/42158694/how-to-run-tensorboard-from-python-scipt-in-virtualenv
def launchTensorBoard():
    os.system('tensorboard --logdir=' + FCN.FLAGS.logs_dir)
    return

t = threading.Thread(target=launchTensorBoard, args=([]))
t.daemon = True
t.start()

print "Ran Tensorboard"
tf.app.run(main=FCN.main)