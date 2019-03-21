TensorFlow
============

TensorFlow is an open source software library for numerical computation using
data flow graphs. Nodes in the graph represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them. This flexible architecture lets you deploy computation to one or
more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code. 

TensorFlow was originally developed by researchers and engineers working on the
Google Brain team within Google's Machine Intelligence research organization
for the purposes of conducting machine learning and deep neural networks
research. The system is general enough to be applicable in a wide variety of
other domains, as well.

## Contents of the TensorFlow image

This container has the TensorFlow Python package installed and ready to use.
`/opt/tensorflow` contains the complete source of this version of TensorFlow.

TensorFlow includes TensorBoard, a data visualization toolkit.

Additionally, this container image also includes several built-in TensorFlow
examples, which can be run using commands like the following:

```
python -m tensorflow.models.image.mnist.convolutional
```
```
python -m tensorflow.models.image.cifar10.cifar10_multi_gpu_train
```

## Running TensorFlow

You can choose to use TensorFlow as provided by NVIDIA, or you can choose to
customize it.

TensorFlow is run simply by importing it as a Python module:
```
$ python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a+b)
42
```

## Customizing TensorFlow

You can customize TensorFlow one of two ways:

(1) Modify the version of the source code in this container and run your
customized version, or (2) use `docker build` to add your customizations on top
of this container if you want to add additional packages.

NVIDIA recommends option 2 for ease of migration to later versions of the
TensorFlow container image.

For more information, see https://docs.docker.com/engine/reference/builder/ for
a syntax reference.  Several example Dockerfiles are provided in the container
image in `/workspace/docker-examples`.

Note that if building additional tensorflow extensions, it is important to match
the build options used to construct the tensorflow libraries provided in this
container. Those options are preserved in /opt/tensorflow/nvbuildopts.

## Suggested Reading

For more information about TensorFlow, including tutorials, documentation, and
examples, see the [TensorFlow tutorials]( https://www.tensorflow.org/tutorials)
and [TensorFlow API] ( https://www.tensorflow.org/api_docs/python).

For information about optimizing TensorFlow models for TensorCore instructions
available on NVIDIA Volta GPUs, see [Training with Mixed Precision](
http://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)


