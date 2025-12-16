#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

echo "========================================="
echo "MEGA-SAM Full Demo Pipeline"
echo "========================================="

evalset=(
  demo
)

DATA_PATH=images/
CKPT_PATH=checkpoints/megasam_final.pth

echo "Step 1/3: Running Mono Depth Demo..."
echo "========================================="

# Run DepthAnything
for seq in ${evalset[@]}; do
  echo "Running DepthAnything for $seq..."
  CUDA_VISIBLE_DEVICES=0 python Depth-Anything/run_videos.py --encoder vitl \
  --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  --img-path $DATA_PATH/$seq \
  --outdir Depth-Anything/video_visualization/$seq
done

# Run UniDepth
export PYTHONPATH="${PYTHONPATH}:$(pwd)/UniDepth"

for seq in ${evalset[@]}; do
  echo "Running UniDepth for $seq..."
  CUDA_VISIBLE_DEVICES=0 python UniDepth/scripts/demo_mega-sam.py \
  --scene-name $seq \
  --img-path $DATA_PATH/$seq \
  --outdir UniDepth/outputs
done

echo "Step 2/3: Running Evaluation Demo..."
echo "========================================="

for seq in ${evalset[@]}; do
    echo "Running evaluation for $seq..."
    CUDA_VISIBLE_DEVICE=0 python camera_tracking_scripts/test_demo.py \
    --datapath=$DATA_PATH/$seq \
    --weights=$CKPT_PATH \
    --scene_name $seq \
    --mono_depth_path $(pwd)/Depth-Anything/video_visualization \
    --metric_depth_path $(pwd)/UniDepth/outputs \
    --disable_vis $@
done

echo "Step 3/3: Running CVD Optimization Demo..."
echo "========================================="

# Run Raft Optical Flows
for seq in ${evalset[@]}; do
  echo "Running RAFT Optical Flow for $seq..."
  CUDA_VISIBLE_DEVICES=0 python cvd_opt/preprocess_flow.py \
  --datapath=$DATA_PATH/$seq \
  --model=cvd_opt/raft-things.pth \
  --scene_name $seq --mixed_precision
done

# Run CVD optmization
for seq in ${evalset[@]}; do
  echo "Running CVD Optimization for $seq..."
  CUDA_VISIBLE_DEVICES=0 python cvd_opt/cvd_opt.py \
  --scene_name $seq \
  --w_grad 2.0 --w_normal 5.0
done

echo "========================================="
echo "Full demo pipeline completed!"
echo "========================================="