
"""
Model definitions for the four networks evaluated in Baumgartner et al.,
"Real-Time Detection and Localisation of Fetal Standard Scan Planes in 2D
Freehand Ultrasound", arXiv preprint:1612.05601 (2016).

Author: Christian Baumgartner (c.f.baumgartner@gmail.com)
Last Update: 14. March 2017
"""

from lasagne.layers import InputLayer, MaxPool2DLayer, Conv2DLayer, batch_norm
from lasagne.layers import GlobalPoolLayer, NonlinearityLayer
from lasagne.nonlinearities import linear, softmax

import theano.tensor as T


def SonoNet16(input_var, image_size, num_labels):

    net = {}
    net['input'] = InputLayer(shape=(None, 1, image_size[0], image_size[1]), input_var=input_var)
    net['conv1_1'] = batch_norm(Conv2DLayer(net['input'], 16, 3, pad=1, flip_filters=False))
    net['conv1_2'] = batch_norm(Conv2DLayer(net['conv1_1'], 16, 3, pad=1, flip_filters=False))
    net['pool1'] = MaxPool2DLayer(net['conv1_2'], 2)

    net['conv2_1'] = batch_norm(Conv2DLayer(net['pool1'], 32, 3, pad=1, flip_filters=False))
    net['conv2_2'] = batch_norm(Conv2DLayer(net['conv2_1'], 32, 3, pad=1, flip_filters=False))
    net['pool2'] = MaxPool2DLayer(net['conv2_2'], 2)

    net['conv3_1'] = batch_norm(Conv2DLayer(net['pool2'], 64, 3, pad=1, flip_filters=False))
    net['conv3_2'] = batch_norm(Conv2DLayer(net['conv3_1'], 64, 3, pad=1, flip_filters=False))
    net['conv3_3'] = batch_norm(Conv2DLayer(net['conv3_2'], 64, 3, pad=1, flip_filters=False))
    net['pool3'] = MaxPool2DLayer(net['conv3_3'], 2)

    net['conv4_1'] = batch_norm(Conv2DLayer(net['pool3'], 128, 3, pad=1, flip_filters=False))
    net['conv4_2'] = batch_norm(Conv2DLayer(net['conv4_1'], 128, 3, pad=1, flip_filters=False))
    net['conv4_3'] = batch_norm(Conv2DLayer(net['conv4_2'], 128, 3, pad=1, flip_filters=False))
    net['pool4'] = MaxPool2DLayer(net['conv4_3'], 2)

    net['conv5_1'] = batch_norm(Conv2DLayer(net['pool4'], 128, 3, pad=1, flip_filters=False))
    net['conv5_2'] = batch_norm(Conv2DLayer(net['conv5_1'], 128, 3, pad=1, flip_filters=False))
    net['conv5_3'] = batch_norm(Conv2DLayer(net['conv5_2'], 128, 3, pad=1, flip_filters=False))

    net['conv5_p'] = batch_norm(Conv2DLayer(net['conv5_3'], num_filters=64, filter_size=(1, 1)))
    net['conv6_p'] = batch_norm(Conv2DLayer(net['conv5_p'], num_filters=num_labels, filter_size=(1, 1), nonlinearity=linear))
    net['average_pool_p'] = GlobalPoolLayer(net['conv6_p'], pool_function=T.mean)
    net['softmax_p'] = NonlinearityLayer(net['average_pool_p'], nonlinearity=softmax)

    net['output'] = net['softmax_p']
    net['feature_maps'] = net['conv6_p']
    net['last_activation'] = net['average_pool_p']

    return net

def SonoNet32(input_var, image_size, num_labels):

    net = {}
    net['input'] = InputLayer(shape=(None, 1, image_size[0], image_size[1]), input_var=input_var)
    net['conv1_1'] = batch_norm(Conv2DLayer(net['input'], 32, 3, pad=1, flip_filters=False))
    net['conv1_2'] = batch_norm(Conv2DLayer(net['conv1_1'], 32, 3, pad=1, flip_filters=False))
    net['pool1'] = MaxPool2DLayer(net['conv1_2'], 2)

    net['conv2_1'] = batch_norm(Conv2DLayer(net['pool1'], 64, 3, pad=1, flip_filters=False))
    net['conv2_2'] = batch_norm(Conv2DLayer(net['conv2_1'], 64, 3, pad=1, flip_filters=False))
    net['pool2'] = MaxPool2DLayer(net['conv2_2'], 2)

    net['conv3_1'] = batch_norm(Conv2DLayer(net['pool2'], 128, 3, pad=1, flip_filters=False))
    net['conv3_2'] = batch_norm(Conv2DLayer(net['conv3_1'], 128, 3, pad=1, flip_filters=False))
    net['conv3_3'] = batch_norm(Conv2DLayer(net['conv3_2'], 128, 3, pad=1, flip_filters=False))
    net['pool3'] = MaxPool2DLayer(net['conv3_3'], 2)

    net['conv4_1'] = batch_norm(Conv2DLayer(net['pool3'], 256, 3, pad=1, flip_filters=False))
    net['conv4_2'] = batch_norm(Conv2DLayer(net['conv4_1'], 256, 3, pad=1, flip_filters=False))
    net['conv4_3'] = batch_norm(Conv2DLayer(net['conv4_2'], 256, 3, pad=1, flip_filters=False))
    net['pool4'] = MaxPool2DLayer(net['conv4_3'], 2)

    net['conv5_1'] = batch_norm(Conv2DLayer(net['pool4'], 256, 3, pad=1, flip_filters=False))
    net['conv5_2'] = batch_norm(Conv2DLayer(net['conv5_1'], 256, 3, pad=1, flip_filters=False))
    net['conv5_3'] = batch_norm(Conv2DLayer(net['conv5_2'], 256, 3, pad=1, flip_filters=False))

    net['conv5_p'] = batch_norm(Conv2DLayer(net['conv5_3'], num_filters=128, filter_size=(1, 1)))
    net['conv6_p'] = batch_norm(Conv2DLayer(net['conv5_p'], num_filters=num_labels, filter_size=(1, 1), nonlinearity=linear))
    net['average_pool_p'] = GlobalPoolLayer(net['conv6_p'], pool_function=T.mean)
    net['softmax_p'] = NonlinearityLayer(net['average_pool_p'], nonlinearity=softmax)

    net['output'] = net['softmax_p']
    net['feature_maps'] = net['conv6_p']
    net['last_activation'] = net['average_pool_p']

    return net

def SonoNet64(input_var, image_size, num_labels):

    net = {}
    net['input'] = InputLayer(shape=(None, 1, image_size[0], image_size[1]), input_var=input_var)
    net['conv1_1'] = batch_norm(Conv2DLayer(net['input'], 64, 3, pad=1, flip_filters=False))
    net['conv1_2'] = batch_norm(Conv2DLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False))
    net['pool1'] = MaxPool2DLayer(net['conv1_2'], 2)

    net['conv2_1'] = batch_norm(Conv2DLayer(net['pool1'], 128, 3, pad=1, flip_filters=False))
    net['conv2_2'] = batch_norm(Conv2DLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False))
    net['pool2'] = MaxPool2DLayer(net['conv2_2'], 2)

    net['conv3_1'] = batch_norm(Conv2DLayer(net['pool2'], 256, 3, pad=1, flip_filters=False))
    net['conv3_2'] = batch_norm(Conv2DLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False))
    net['conv3_3'] = batch_norm(Conv2DLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False))
    net['pool3'] = MaxPool2DLayer(net['conv3_3'], 2)

    net['conv4_1'] = batch_norm(Conv2DLayer(net['pool3'], 512, 3, pad=1, flip_filters=False))
    net['conv4_2'] = batch_norm(Conv2DLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False))
    net['conv4_3'] = batch_norm(Conv2DLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False))
    net['pool4'] = MaxPool2DLayer(net['conv4_3'], 2)

    net['conv5_1'] = batch_norm(Conv2DLayer(net['pool4'], 512, 3, pad=1, flip_filters=False))
    net['conv5_2'] = batch_norm(Conv2DLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False))
    net['conv5_3'] = batch_norm(Conv2DLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False))

    net['conv5_p'] = batch_norm(Conv2DLayer(net['conv5_3'], num_filters=256, filter_size=(1, 1)))
    net['conv6_p'] = batch_norm(Conv2DLayer(net['conv5_p'], num_filters=num_labels, filter_size=(1, 1), nonlinearity=linear))
    net['average_pool_p'] = GlobalPoolLayer(net['conv6_p'], pool_function=T.mean)
    net['softmax_p'] = NonlinearityLayer(net['average_pool_p'], nonlinearity=softmax)

    net['output'] = net['softmax_p']
    net['feature_maps'] = net['conv6_p']
    net['last_activation'] = net['average_pool_p']

    return net

def SmallNet(input_var, image_size, num_labels):

    net = {}
    net['input'] = InputLayer(shape=(None, 1, image_size[0], image_size[1]), input_var=input_var)
    net['conv1'] = Conv2DLayer(net['input'], num_filters=32, filter_size=(7, 7), stride=(2, 2))
    net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=(2, 2))
    net['conv2'] = Conv2DLayer(net['pool1'], num_filters=64, filter_size=(5, 5), stride=(2, 2))
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=(2, 2))
    net['conv3'] = Conv2DLayer(net['pool2'], num_filters=128, filter_size=(3, 3), pad=(1, 1))
    net['conv4'] = Conv2DLayer(net['conv3'], num_filters=128, filter_size=(3, 3), pad=(1, 1))

    net['conv5_p'] = Conv2DLayer(net['conv4'], num_filters=64, filter_size=(1, 1))
    net['conv6_p'] = Conv2DLayer(net['conv5_p'], num_filters=num_labels, filter_size=(1, 1), nonlinearity=linear)
    net['average_pool_p'] = GlobalPoolLayer(net['conv6_p'], pool_function=T.mean)
    net['softmax_p'] = NonlinearityLayer(net['average_pool_p'], nonlinearity=softmax)

    net['output'] = net['softmax_p']
    net['feature_maps'] = net['conv6_p']
    net['last_activation'] = net['average_pool_p']

    return net
