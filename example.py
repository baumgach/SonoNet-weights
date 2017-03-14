
"""
Usage example for models defined in models.py. Change the variable 'network_name'
to evaluate the other networks. By default SonoNet32 is evaluated. The example
images used for this script are from the test set from experiment III-A in

"Real-Time Detection and Localisation of Fetal Standard Scan Planes in 2D
Freehand Ultrasound", arXiv preprint:1612.05601 (2016).

Author: Christian Baumgartner (c.f.baumgartner@gmail.com)
Last Update: 14. March 2017
"""

import lasagne
import theano
import theano.tensor as T
import numpy as np

import glob
from scipy.misc import imread, imresize, imshow
import time

import models

### CONSTANTS ###

# network used for evaluation:
# network_name takes values in {'SmallNet','SonoNet16', 'SonoNet32', 'SonoNet64'}
network_name = 'SonoNet32'

# The mapping from network output to label name
label_names = [ '3VV',
                '4CH',
                'Abdominal',
                'Background',
                'Brain (Cb.)',
                'Brain (Tv.)',
                'Femur',
                'Kidneys',
                'Lips',
                'LVOT',
                'Profile',
                'RVOT',
                'Spine (cor.)',
                'Spine (sag.) ']

# Crop range used to get rid of the vendor info etc around the images
crop_range = [(115, 734), (81, 874)]  # [(top, bottom), (left, right)]

# The input images will be resized to this size
input_size = [224, 288]

# Display the images during the prediction
display_images = True

### HELPER FUNCTIONS ###

def read_model(layer, filename):
    """ Load the weights of a network """
    with np.load(filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(layer, param_values)

def imcrop(image, crop_range):
    """ Crop an image to a crop range """
    return image[crop_range[0][0]:crop_range[0][1],
                 crop_range[1][0]:crop_range[1][1], ...]

if __name__ == "__main__":

    ### LOAD IMAGES ###

    input_list = []

    for filename in glob.glob('example_images/*.tiff'):

        # prepare images
        image = imread(filename)  # read
        image = imcrop(image, crop_range)  # crop
        image = imresize(image, input_size)  # resize
        image = np.mean(image,axis=2)  # convert to gray scale

        # convert to 4D tensor of type float32
        image_data = np.float32(np.reshape(image,
                                          (1,1,image.shape[0], image.shape[1])))

        # normalise images by substracting mean and dividing by standard dev.
        mean = image_data.mean()
        std = image_data.std()
        image_data = np.array(255.0*np.divide(image_data - mean, std),
                              dtype=np.float32)
                              # Note that the 255.0 scale factor is arbitrary
                              # it is necessary because the network was trained
                              # like this, but the same results would have been
                              # achieved without this factor for training.

        input_list.append(image_data)

    ### PREPARE NETWORK ###

    # input tensors
    input_var = T.tensor4('inputs')

    # Defining the model and reading the paramters
    network_builder = getattr(models,network_name)
    net = network_builder(input_var, input_size, num_labels=len(label_names))
    read_model(net['output'], '%s.npz' % network_name)

    # Defining the predictino function
    prediction_var = lasagne.layers.get_output(net['output'], deterministic=True)
    pred_and_conf_fn = theano.function(
                            [input_var],
                            [T.argmax(prediction_var, axis=1),
                             T.max(prediction_var, axis=1)]
                            )

    ### RUN PREDICTIONS ###

    total_time = 0  # measures the total time spent predicting in seconds

    print "\nPredictions using %s:" % network_name

    for X, file_name in zip(input_list, glob.glob('example_images/*.tiff')):

        start_time = time.time()
        [prediction, confidence] = pred_and_conf_fn(X)  # get the prediction
        total_time += time.time() - start_time

        true_label = file_name.split('/')[1].split('.')[0]

        # True labels are obtained from file name.
        print " - %s (conf: %.2f, true label: %s)" % (label_names[prediction[0]],
                                                      confidence[0],
                                                      true_label)

        if display_images:
            imshow(np.squeeze(X))

    print "Average FPS: %.2f" % (float(len(input_list))/total_time)
