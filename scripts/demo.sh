#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH -A YOUR_PROJECT
#SBATCH -t 00:20:00
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -p gpu
#SBATCH --gpus:1

YOLO_CHECKPOINT_FILE="..." # path to the head detector checkpoint you want to use

TEMPORAL_CONTEXT=0 # demo code uses the static model
MTGS_CHECKPOINT_FILE="..." # path to the MTGS checkpoint you want to use

VIDEO_FILE="..." # path to the video file you want to run the demo on
OUTPUT_FOLDER="..." # path to the folder where you want to save the demo outputs


python ./demo.py head_detector.checkpoint_file=$YOLO_CHECKPOINT_FILE demo.video_file=$VIDEO_FILE demo.checkpoint_file=$MTGS_CHECKPOINT_FILE demo.output_folder=$OUTPUT_FOLDER data.temporal_context=$TEMPORAL_CONTEXT 
