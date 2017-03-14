# SonoNet weights

This repository contains pretrained weights and model descriptions for all of
the SonoNet variations described in our recent submission:

Baumgartner et al., "Real-Time Detection and Localisation of Fetal Standard Scan Planes in 2D
Freehand Ultrasound", arXiv preprint:1612.05601 (2016),

and prior work published here:

Baumgartner et al., "Real-time standard scan plane detection and localisation in fetal ultrasound
using fully convolutional neural networks", Proc. MICCAI (2016).

Please acknowledge the first of the two papers above if you end up using the
these weights for your work.

The networks were trained using [theano](http://deeplearning.net/software/theano/)
and the [lasagne](https://github.com/Lasagne/Lasagne) deep learning framework.

The weights are saved in the respective `.npz` files, the model definitions are
given in `models.py`. A minimal example for classifying the images in the folder
`example_images` is given in `example.py`.

## Setup

Running `example.py` requires the latest theano and lasagne versions. Follow
the instructions [here](http://lasagne.readthedocs.io/en/latest/user/installation.html),
under the section "Bleeding Edge".

Furthermore, `numpy`, `scipy` are required.
