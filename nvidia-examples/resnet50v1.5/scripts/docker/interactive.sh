#!/bin/bash

nvidia-docker run -it --rm --ipc=host -v $PWD:/workspace/rn50v15_tf/ rn50v15_tf bash
