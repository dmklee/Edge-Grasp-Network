## Install
```
pip install numpy open3d pybullet
```

## Generate Data
```
python clutter_grasp_data_generator.py --object_set=<object_set>
```
Select object set from "berkeley\_adversarial" and "graspnet1B".

## Acknowledgement
Code based on [Edge Grasp Repo](https://github.com/haojhuang/Edge-Grasp-Network/). Berkeley Adversarial objects taken from [here](http://bit.ly/3ViL0ha) and then
convexified with `pybullet.vhacd`.

