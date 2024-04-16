#!/bin/bash

OBJECT_SET=graspnet1B
NUM_GRASPS=100000
NUM_OBJECTS=1
SEED=1
NUM_PROCS=20

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
    --dont_balance_successes --dest=/home/jpark_theaiinstitute_com/git/PC_FM/Edge-Grasp-Network/graspnet_obj_classification_train \
    --attempts_per_scene=1 --seed=$((SEED + i)) &
  pids+=("$!")
done

wait
