#!/bin/bash

OBJECT_SET=graspnet1B
NUM_GRASPS=5000
NUM_OBJECTS=1
SEED=2021
NUM_PROCS=10

grasps_per_proc=$((NUM_GRASPS / NUM_PROCS))

pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    kill -0 "$pid" && kill "$pid"
  done
}
trap cleanup EXIT TERM

for i in $(seq ${NUM_PROCS}); do
  python clutter_grasp_data_generator.py --object_set=$OBJECT_SET \
    --num_grasps=$grasps_per_proc --num_objects=$NUM_OBJECTS \
    --dest=/home/jpark_theaiinstitute_com/git/PC_FM/pc_fm/data/grasping3cam_allobjs_50k/val \
    --dont_balance_successes --attempts_per_scene=1 --seed=$((SEED + i)) &
  pids+=("$!")
done

wait
