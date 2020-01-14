#!/bin/bash
sudo docker run -it --gpus all --name devtest \
   --mount type=bind,source="$(pwd)",target=/bluefog bluefog:devel
