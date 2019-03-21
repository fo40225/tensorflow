# ResNet50 v1.5

## The model
The ResNet50 v1.5 model is a modified version of the [original ResNet50 v1 model](https://arxiv.org/abs/1512.03385).

The difference between v1 and v1.5 is in the bottleneck blocks which requires
downsampling, for example, v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution.

This difference makes ResNet50 v1.5 slightly more accurate (~0.5% top1) than v1,
but comes with a small performance drawback (~5% imgs/sec).

The following features were implemented in this model:
* data-parallel multi-GPU training with Horovod
* Tensor Cores (mixed precision) training
* static loss scaling for Tensor Cores (mixed precision) training

The following performance optimizations were implemented in this model:
* XLA support (experimental)

## Training procedure

### Optimizer

This model trains for 90 epochs, with standard ResNet v1.5 setup:

* SGD with momentum (0.9)

* Learning rate = 0.1 for 256 batch size, for other batch sizes we linearly
scale the learning rate.

* Learning rate decay - multiply by 0.1 after 30, 60, and 80 epochs

* For bigger batch sizes (512 and up) we use linear warmup of the learning rate
during first 5 epochs according to [Training ImageNet in 1 hour](https://arxiv.org/abs/1706.02677).

* Weight decay: 1e-4


### Data Augmentation

This model uses the following data augmentation:

* During training, we perform the following augmentation techniques:
  * Normalization
  * Random resized crop to 224x224
    * Scale from 8% to 100%
    * Aspect ratio from 3/4 to 4/3
  * Random horizontal flip

* During inference, we perform the following augmentation techniques:
  * Normalization
  * Scale to 256x256
  * Center crop to 224x224


### Other training recipes

This script does not target any specific benchmark.
There are changes that others have made which can speed up convergence and/or increase accuracy.

One of the more popular training recipes is provided by [fast.ai](https://github.com/fastai/imagenet-fast).

The fast.ai recipe introduces many changes to the training procedure, one of which is progressive resizing of the training images.

The first part of training uses 128px images, the middle part uses 224px images, and the last part uses 288px images.
The final validation is performed on 288px images.

The training script in this repository performs validation on 224px images, just like the original paper described.

These two approaches can't be directly compared, since the fast.ai recipe requires validation on 288px images,
and this recipe keeps the original assumption that validation is done on 224px images.

Using 288px images means that a lot more FLOPs are needed during inference to reach the same accuracy.

# Setup

The following section list the requirements that you need to meet in order to use the ResNet50 v1.5 model.

## Requirements
This repository contains Dockerfile which extends the Tensorflow NGC container and encapsulates all dependencies.  Aside from these dependencies, ensure you have the following software:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [TensorFlow 19.02-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
* (optional) NVIDIA Volta GPU (see section below) - for best training performance using mixed precision

For more information about how to get started with NGC containers, see the
following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
[Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html),
[Accessing And Pulling From The NGC container registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
[Running Tensorflow](https://docs.nvidia.com/deeplearning/dgx/tensorflow-release-notes/running.html#running).

# Training
## Training using mixed precision with Tensor Cores

Before you can train using mixed precision with Tensor Cores, ensure that you have a
[NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)-based GPU. Other platforms might likely work but aren't officially supported. 

For information about how to train using mixed precision, see the
[Mixed Precision Training paper](https://arxiv.org/abs/1710.03740)
and
[Training With Mixed Precision documentation](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html).


# Quick start guide
In order to launch the training using the default parameters of the ResNet50 model on the ImageNet dataset, perform the following steps:

## 1. Download and preprocess the dataset
The ResNet50 v1.5 script operates on ImageNet 1k, a widely popular image classification dataset from ILSVRC challenge.
To download and preprocess the dataset, use the [Generate ImageNet for TensorFlow](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_and_preprocess_imagenet.sh) script. The dataset will be downloaded to a directory specified as the first parameter of the script.



## 2. Build and launch the Docker container:
```bash
bash scripts/docker/build.sh
```

```bash
bash scripts/docker/interactive.sh
```

## 3. Run training  
To run training for a standard configuration (1/4/8 GPUs, FP16/FP32), run one of the scripts in the `./scripts` directory
called `./scripts/RN50_{FP16, FP32}_{1, 4, 8}GPU.sh`. Each of the scripts requries three parameters: path to the root directory of model as first argument, path to dataset as a second and results destination - as third argument:

```bash
./scripts/RN50_FP16_8GPU.sh <path to model> <path to dataset> <path to results>
```


To run a non-standard configuration, use:

* For 1 GPU
    * FP32
        `python ./main.py --precision=fp32 --batch_size=128 --data_dir=<path to imagenet> --results_dir=<path to results>`
    * FP16
        `python ./main.py --precision=fp16 --batch_size=256 --data_dir=<path to imagenet> --results_dir=<path to results>`

* For multiple GPUs
    * FP32
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --precision=fp32 --batch_size=128 --data_dir=<path to imagenet> --results_dir=<path to results>`
    * FP16
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --precision=fp16 --batch_size=256 --data_dir=<path to imagenet> --results_dir=<path to results>`

Use `python ./main.py -h` to obtain the list of available options in the `main.py` script.

## 4. Run evaluation

To run evaluation on a checkpointed model, run one of the scripts in the `./scripts` directory
called `./scripts/RN50_{FP16, FP32}_EVAL.sh`. Each of the scripts requries three parameters: path to the root directory of model as first argument, path to dataset as a second and results destination - as third argument:

```bash
./scripts/RN50_FP16_EVAL.sh <path to model> <path to dataset> <path to results>
```


To run a non standard configuration, use:

`python ./main.py --mode=evaluate --precision=<precision> --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to checkpoint>`

## Benchmarking
### Training performance benchmark

To benchmark the training performance on a specific batch size, run:

* For 1 GPU
    * FP32
        `python ./main.py --mode=training_benchmark --precision=fp16 --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * FP16
        `python ./main.py --mode=training_benchmark --precision=fp32 --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
* For multiple GPUs
    * FP32
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --mode=training_benchmark --precision=fp32 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * FP16
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --mode=training_benchmark --precision=fp16 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
        
Each of these scripts runs 200 warm-up iterations and measures the first epoch.

To control warmup and benchmark length, use `--warmup_steps`, `--num_iter` and `--iter_unit` flags.


### Inference performance and accuracy benchmark

To benchmark the inference performance on a specific batch size, run:

* FP32
`python ./main.py --mode=inference_benchmark --precision=fp32 --warmup_steps 20 --train_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --log_dir=<path to results directory>`

* FP16
`python ./main.py --mode=inference_benchmark --precision=fp16 --warmup_steps 20 --train_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --log_dir=<path to results directory>`

Each of these scripts, by default runs 20 warm-up iterations and measures the next 80 iterations.

To control warm-up and benchmark length, use `--warmup_steps`, `--num_iter` and `--iter_unit` flags.

# Results

The following sections provide details on how we achieved our results in training, performance and inference performance.

## Training accuracy results

Our results were obtained by running the `./scripts/RN50_{FP16, FP32}_{1, 4, 8}GPU.sh` script in
the tensorflow-19.02-py3 Docker container on NVIDIA DGX-1 with 8 V100 16G GPUs.


| **number of GPUs** | **mixed precision top1** | **mixed precision training time** | **FP32 top1** | **FP32 training time** |
|:------------------:|:------------------------:|:---------------------------------:|:-------------:|:----------------------:|
| **1**                  | 76.32                    | 41.3h                             | 76.11         | 89.4h                  |
| **4**                  | 76.35                    | 10.5h                             | 76.58         | 22.4h                  |
| **8**                  | 76.31                    | 5.6h                              | 76.31         | 11.5h                  |



## Training performance results

Our results were obtained by running the `./scripts/benchmarking/DGX1V_trainbench_fp16.sh` and `./scripts/benchmarking/DGX1V_trainbench_fp32.sh` scripts in the tensorflow-19.02-py3 Docker container on NVIDIA DGX-1 with 8 V100 16G GPUs.


| **number of GPUs** | **mixed precision img/s** | **FP32 img/s** | **mixed precision speedup** | **mixed precision weak scaling** | **FP32 weak scaling** |
|:------------------:|:-------------------------:|:--------------:|:---------------------------:|:--------------------------------:|:---------------------:|
| **1**                  | 778.0                     | 359.5          | 2.16                        | 1.00                             | 1.00                  |
| **4**                  | 3079.6                    | 1405.4         | 2.19                        | 3.96                             | 3.91                  |
| **8**                  | 6115.4                    | 2768.2         | 2.21                        | 7.86                             | 7.70                  |

Our results were obtained by running the Our were obtained by running the `./scripts/benchmarking/DGX1V_inferbench_fp16.sh` and `./scripts/benchmarking/DGX1V_inferbench_fp32.sh` scripts in the tensorflow-19.02-py3 Docker container on NVIDIA DGX-1 with 8 V100 16G GPUs.


## Inference performance results

| **batch size** | **mixed precision img/s** | **FP32 img/s** |
|:--------------:|:-------------------------:|:--------------:|
|         **1** |   177.2 |   170.8 |      
|         **2** |   325.7 |   308.4 |
|         **4** |   587.0 |   499.4 |         
|         **8** |  1002.9 |   688.3 |         
|        **16** |  1408.5 |   854.9 |        
|        **32** |  1687.0 |   964.4 |        
|        **64** |  1907.7 |  1045.1 |
|       **128** |  2077.3 |  1100.1 |       
|       **256** |  2129.3 |  N/A    |



# Changelog
1. March 1, 2019
  * Initial release

# Known issues
There are no known issues with this model.
