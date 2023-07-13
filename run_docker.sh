#!/bin/sh

sudo docker run -it --rm --gpus=all --shm-size=32G -p 6006:6006 -p 8080:8080 \
                -v /workspace/:/workspace/ -v /mnt/:/mnt/ -v /media/:/media/ \
                -w /workspace/ panorama_depth_docker:latest bash
