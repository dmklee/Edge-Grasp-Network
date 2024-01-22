import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from simulator.simulation_clutter_bandit import ClutterRemovalSim
from simulator.perception import camera_on_sphere, create_tsdf

from simulator.transform import Transform, Rotation
import warnings


def render_images(sim: ClutterRemovalSim, n: int, rng: np.random.RandomState):
    '''Render depth images from random views
    '''
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0 + 0.25])
    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        r = rng.uniform(1.5, 2.5) * sim.size
        theta = rng.uniform(np.pi/4, np.pi/3)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]
        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img
        eye = np.r_[
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
        eye = eye + origin.translation

    return depth_imgs, extrinsics, eye


def main(args):
    rng = np.random.RandomState(args.seed)
    root = Path(args.dest) / args.object_set

    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.gui, rng=rng)

    num_grasps = 0
    scene_id = 0
    while num_grasps < args.num_grasps:
        object_count = rng.poisson(args.num_objects) + 1
        sim.reset(object_count)
        sim.save_state()

        # render point clouds
        n = rng.choice(a=[1, 2, 3], p=[0.6, 0.2, 0.2])
        depth_imgs, extrinsics, eye = render_images(sim, n, rng)

        # reconstrct point cloud using a subset of the images
        tsdf = create_tsdf(sim.size, 180, depth_imgs, sim.camera.intrinsic, extrinsics)
        pc = tsdf.get_cloud()
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
        pc = pc.crop(bounding_box)
        # o3d.visualization.draw_geometries([pc])

        if pc.is_empty():
            print("Empty point cloud, skipping scene")
            continue

        # clean up the point cloud
        vertices = np.asarray(pc.points)
        # add gaussian noise 95% confident interval (-1.96,1.96)
        vertices = vertices + rng.normal(loc=0.0,scale=0.0005, size=(len(vertices),3))
        pc.points = o3d.utility.Vector3dVector(vertices)

        pc, ind = pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
        pc, ind = pc.remove_radius_outlier(nb_points=30, radius=0.03)
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pc.orient_normals_consistent_tangent_plane(20)

        vertices = np.asarray(pc.points)
        if len(vertices) < 300:
            continue
        pc = pc.voxel_down_sample(voxel_size=0.0045)

        poses = np.eye(4)[None]
        # sample position in collision and random orientation
        obj_centers = np.array(
            [sim.world.bodies[i].get_pose().translation for i in range(1, 1+sim.num_objects)]
        )
        obj_ids = rng.randint(len(obj_centers), size=args.attempts_per_scene)

        pos = obj_centers[obj_ids] + rng.normal(loc=0, scale=0.015, size=(len(obj_ids), 3))
        euler = rng.uniform([0, 0, 0],[2*np.pi, 0.4*np.pi, 2*np.pi], size=(len(obj_ids), 3))

        poses = np.zeros((len(obj_ids), 4, 4))
        poses[:, 3, 3] = 1.
        poses[:, :3, 3] = pos
        poses[:, :3, :3] = (
            Rotation.from_euler('y', np.pi) * Rotation.from_euler('zyz', euler)
        ).as_matrix()

        labels = np.zeros(len(poses), dtype=bool)
        widths = np.zeros(len(poses), dtype=np.float32)
        for i, pose in enumerate(poses):
            sim.restore_state()
            label, width = sim.execute_grasp(Transform.from_matrix(pose), remove=False)
            labels[i] = label
            widths[i] = width

        sr = np.mean(labels)

        # balance to 50% success/failure
        success_ids = np.argwhere(labels == True).flatten()
        failure_ids = np.argwhere(labels == False).flatten()
        failure_ids = failure_ids[:len(success_ids)]

        labels = labels[np.concatenate([success_ids, failure_ids])]
        poses = poses[np.concatenate([success_ids, failure_ids])]
        num_grasps += len(poses)

        scene_path = root / f"{scene_id:03d}"
        scene_path.mkdir(parents=True, exist_ok=True)
        np.save(str(scene_path / 'pc.npy'), np.asarray(pc.points).astype(np.float32))
        np.save(str(scene_path / 'labels.npy'), labels)
        np.save(str(scene_path / 'poses.npy'), poses)

        print(f"Scene_{scene_id}: SR={sr:.2%} | Total={num_grasps}")
        scene_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default="./grasp_data")
    parser.add_argument("--num_grasps", type=int, default=1000)
    parser.add_argument("--attempts_per_scene", type=int, default=100)
    parser.add_argument("--num_objects", type=int, default=4)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="packed")
    parser.add_argument("--object_set", type=str, default="berkeley_adversarial")
    parser.add_argument("--gui", action="store_true", default= False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
