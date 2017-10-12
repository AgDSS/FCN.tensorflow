"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import tensorflow as tf

class BatchDatset:
    files = []
    images = np.array([])
    image_arr = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        image_augmentation=True/False
        :param predict_dataset: boolean stating whether dataset is for predictions (does not include annotations)
            True/False (default False)
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.__channels = True
        self.image_arr = [self._transform(filename['image']) for filename in self.files]
        if self.image_options.get("image_augmentation", False):
            self.images = np.array([self._augment_image(image) for image in self.image_arr])
        else:
            self.images = np.array(self.image_arr)
        self.__channels = False
        print (self.images.shape)
        if not self.image_options.get("predict_dataset", False):
            self.annotations = np.array(
                [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
            print (self.annotations.shape)

    def _augment_image(self, image, annotation_file=None):
        if annotation_file:
            combined_image_label = np.concat(image, annotation_file, axis=2)
            combined_image_label = tf.image.random_flip_left_right(combined_image_label)
            distorted_image = combined_image_label[:, :, :3]
            distorted_annotation = combined_image_label[:, :, :3]
        else:
            distorted_image = tf.image.random_flip_left_right(image)
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
        if annotation_file:
            return distorted_image, distorted_annotation
        else:
            return distorted_image

    def _transform(self, filename):
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        if not self.image_options.get("predict_dataset", False):
            return self.images, self.annotations
        else:
            return self.images

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            if self.image_options.get("image_augmentation", False):
                self.images = np.array([self._augment_image(image) for image in self.image_arr])
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            if not self.image_options.get("predict_dataset", False):
                self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        if not self.image_options.get("predict_dataset", False):
            return self.images[start:end], self.annotations[start:end]
        else:
            return self.images[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        if not self.image_options.get("predict_dataset", False):
            return self.images[indexes], self.annotations[indexes]
        else:
            return self.images[indexes]


