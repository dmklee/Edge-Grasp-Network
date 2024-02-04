## Install
```
pip install numpy open3d pybullet
```

## Generate Data
Generate 10K grasps for training, either with all objects from graspnet1B (`graspnet1B`) or 80% of 
objects from graspnet1B (`graspnet1B-train`).
```
python clutter_grasp_data_generator.py --object_set=graspnet1B --num_grasps=10000 --num_objects=1 --seed=0
python clutter_grasp_data_generator.py --object_set=graspnet1B-train --num_grasps=10000 --num_objects=1 --seed=1
```
Generate 1K grasps for evaluation, either with 20% of objects from graspnet1B (`graspnet1B-val`) or
adversarial objects (`berkeley_adversarial`).
```
python clutter_grasp_data_generator.py --object_set=graspnet1B-val --num_grasps=1000 --num_objects=1 --seed=2
python clutter_grasp_data_generator.py --object_set=berkeley_adversarial --num_grasps=1000 --num_objects=1 --seed=2
```

## Acknowledgement
Code based on [Edge Grasp Repo](https://github.com/haojhuang/Edge-Grasp-Network/). Berkeley Adversarial objects taken from [here](http://bit.ly/3ViL0ha) and then
convexified with `pybullet.vhacd`.

