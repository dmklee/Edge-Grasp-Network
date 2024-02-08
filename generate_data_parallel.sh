#!/bin/bash

OBJECT_SET=graspnet1B
NUM_GRASPS=1000000
NUM_OBJECTS=1
SEED=1
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
    --seed=$((SEED + i)) &
  pids+=("$!")
done

wait
