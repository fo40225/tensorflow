# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import warnings
warnings.simplefilter("ignore")

import tensorflow as tf

import horovod.tensorflow as hvd
from utils import hvd_utils

from runtime import Runner

from utils.cmdline_helper import parse_cmdline

if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)

    FLAGS = parse_cmdline()

    RUNNING_CONFIG = tf.contrib.training.HParams(
        mode=FLAGS.mode,

        # ======= Directory HParams ======= #
        log_dir=FLAGS.results_dir,
        model_dir=FLAGS.results_dir,
        summaries_dir=FLAGS.results_dir,
        data_dir=FLAGS.data_dir,

        # ========= Model HParams ========= #
        n_classes=1001,
        input_format='NHWC',
        compute_format=FLAGS.data_format,
        dtype=tf.float32 if FLAGS.precision == "fp32" else tf.float16,
        height=224,
        width=224,
        n_channels=3,

        # ======= Training HParams ======== #
        iter_unit=FLAGS.iter_unit,
        num_iter=FLAGS.num_iter,
        warmup_steps=FLAGS.warmup_steps,
        batch_size=FLAGS.batch_size,
        log_every_n_steps=FLAGS.display_every,
        learning_rate_init=FLAGS.lr_init,
        weight_decay=FLAGS.weight_decay,
        momentum=FLAGS.momentum,
        loss_scale=FLAGS.loss_scale,
        allow_xla=FLAGS.allow_xla,
        distort_colors=False,
        seed=FLAGS.seed,
    )

    # ===================================

    runner = Runner(
        n_classes=RUNNING_CONFIG.n_classes,
        input_format=RUNNING_CONFIG.input_format,
        compute_format=RUNNING_CONFIG.compute_format,
        dtype=RUNNING_CONFIG.dtype,
        height=RUNNING_CONFIG.height,
        width=RUNNING_CONFIG.width,
        n_channels=RUNNING_CONFIG.n_channels,
        distort_colors=RUNNING_CONFIG.distort_colors,
        log_dir=RUNNING_CONFIG.log_dir,
        model_dir=RUNNING_CONFIG.model_dir,
        data_dir=RUNNING_CONFIG.data_dir,
        seed=RUNNING_CONFIG.seed
    )

    if RUNNING_CONFIG.mode in ["train", "train_and_evaluate", "training_benchmark"]:

        runner.train(
            iter_unit=RUNNING_CONFIG.iter_unit,
            num_iter=RUNNING_CONFIG.num_iter,
            batch_size=RUNNING_CONFIG.batch_size,
            warmup_steps=RUNNING_CONFIG.warmup_steps,
            log_every_n_steps=RUNNING_CONFIG.log_every_n_steps,
            weight_decay=RUNNING_CONFIG.weight_decay,
            learning_rate_init=RUNNING_CONFIG.learning_rate_init,
            momentum=RUNNING_CONFIG.momentum,
            loss_scale=RUNNING_CONFIG.loss_scale,
            allow_xla=RUNNING_CONFIG.allow_xla,
            is_benchmark=RUNNING_CONFIG.mode == 'training_benchmark'
        )

    if RUNNING_CONFIG.mode in ["train_and_evaluate", 'evaluate', 'inference_benchmark']:

        if RUNNING_CONFIG.mode == 'inference_benchmark' and hvd_utils.is_using_hvd():
            raise NotImplementedError("Only single GPU inference is implemented.")

        elif not hvd_utils.is_using_hvd() or hvd.rank() == 0:

            runner.evaluate(
                iter_unit=RUNNING_CONFIG.iter_unit if RUNNING_CONFIG.mode != "train_and_evaluate" else "epoch",
                num_iter=RUNNING_CONFIG.num_iter if RUNNING_CONFIG.mode != "train_and_evaluate" else 1,
                warmup_steps=RUNNING_CONFIG.warmup_steps,
                batch_size=RUNNING_CONFIG.batch_size,
                log_every_n_steps=RUNNING_CONFIG.log_every_n_steps,
                allow_xla=RUNNING_CONFIG.allow_xla,
                is_benchmark=RUNNING_CONFIG.mode == 'inference_benchmark'
            )
