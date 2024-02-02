import os
import time
import sys
from pathlib import Path
import numpy as np
import pybullet
from simulator.perception import CameraIntrinsic, TSDFVolume, camera_on_sphere
from simulator.btsim import BtWorld
from simulator.transform import Rotation, Transform
from scipy.spatial.transform import Slerp


class ClutterRemovalSim(object):
    def __init__(self, scene, obj_folder, gui=True, rng=None):
        assert scene in ["pile", "packed"]
        self.urdf_root = Path("./data_robot/urdfs")
        if '-' in obj_folder:
            obj_folder, mode = obj_folder.split('-')
            self.obj_root = Path('./data_robot') / obj_folder
            obj_files = list(sorted(self.obj_root.glob('*.obj')))
            L = int(0.8*len(obj_files))
            self.obj_files = obj_files[:L] if mode=='train' else obj_files[L:]
        else:
            self.obj_root = Path('./data_robot') / obj_folder
            self.obj_files = list(sorted(self.obj_root.glob('*.obj')))

        self.scene = scene

        # get the list of urdf files or obj files
        self.global_scaling = 1.2 if obj_folder == 'berkeley_adversarial' else 1.2
        self.gui = gui
        self.rng = rng or np.random.RandomState()
        self.world = BtWorld(self.gui)
        self.gripper = Gripper(self.world)
        self.size = 6 * self.gripper.finger_depth
        intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count, index=None):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        #self.world.p.configureDebugVisualizer(self.world.p.COV_ENABLE_GUI, 0)
        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)

        if self.scene == "pile":
            self.generate_pile_scene(object_count, table_height,True, index=index)

        elif self.scene == "packed":
            self.generate_packed_scene(object_count, table_height)

        else:
            raise ValueError("Invalid scene argument")

    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6,table=True)

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height, self.size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]

    def generate_pile_scene(self, object_count, table_height, return_urdf=False,index=None):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3,table=True)
        # drop objects
        if index is not None:
            objs = [self.obj_files[index]]
        else:
            objs = self.rng.choice(self.obj_files, size=object_count)

        #print(urdfs)
        for obj in objs:
            rotation = Rotation.random(random_state=self.rng)

            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = self.rng.uniform(0.8, 1.0)
            self.world.load_obj(obj, pose, scale=self.global_scaling*scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        # remove box
        self.world.remove_body(box)
        self.remove_and_wait()

    def generate_packed_scene(self, object_count, table_height):
        attempts = 0
        max_attempts = 12
        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            obj = self.rng.choice(self.obj_files)

            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 1.0
            if self.rng.random() < 0.3:
                rotation = Rotation.from_euler('xz', (-np.pi/2, self.rng.uniform(0, 2*np.pi)))
            else:
                rotation = Rotation.random(random_state=self.rng)
            pose = Transform(rotation, np.r_[x, y, z])
            scale = self.rng.uniform(0.8, 1.0)
            body = self.world.load_obj(obj, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002

            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1

    def acquire_tsdf(self, n, N=None):
        """Render synthetic depth images from n viewpoints and integrate into a TSDF.
        If N is None, the n viewpoints are equally distributed on circular trajectory.
        If N is given, the first n viewpoints on a circular trajectory consisting of N points are rendered.
        """
        tsdf = TSDFVolume(self.size, 40)
        high_res_tsdf = TSDFVolume(self.size, 120)

        origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0])
        r = 2.0 * self.size
        theta = np.pi / 6.0

        N = N if N else n
        phi_list = 2.0 * np.pi * np.arange(n) / N
        extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]
        timing = 0.0
        for extrinsic in extrinsics:
            depth_img = self.camera.render(extrinsic)[1]
            tic = time.time()
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            timing += time.time() - tic
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)

        return tsdf, high_res_tsdf.get_cloud(), timing

    def rotate(self, theta, eef_step=0.05, vel=0.80, axis='z'):
        # eef_step=0.05, vel=0.40,
        T_world_body = self.gripper.body.get_pose()
        T_world_tcp = T_world_body * self.gripper.T_body_tcp
        #pre_position = T_world_tcp.translation
        diff = theta
        n_step = int(abs(theta) / eef_step)
        if n_step == 0:
            n_step = 1  # avoid divide by zero
        dist_step = diff / n_step
        dur_step = abs(dist_step) / vel
        for _ in range(n_step):
            # T_world_tcp = Transform(Rotation.from_euler(axis,dist_step),[0.0,0.0,0.0]) * T_world_tcp
            T_world_tcp = T_world_tcp * Transform(Rotation.from_euler(axis, -dist_step), [0.0, 0.0, 0.0])
            self.gripper.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()

    def advance_sim(self,frames):
        for _ in range(frames):
            self.world.step()

    def gripper_dance(self,n_rotations=9):
        #center_pose = grasp.pose
        #yaws = np.linspace(0.0, np.pi, 9, endpoint=False)
        yaw = np.pi/np.float(n_rotations)
        for _ in range(n_rotations):
            self.rotate(yaw,axis='y')


    def execute_grasp(self, grasp_pose: Transform, remove=True, allow_contact=True):

        pregrasp_pose = grasp_pose * Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        retreat_pose = Transform(Rotation.identity(), [0.0, 0.0, 0.2]) * grasp_pose

        self.gripper.reset(pregrasp_pose)
        if self.gripper.detect_contact():
            result = False, self.gripper.max_opening_width

        else:
            self.gripper.move_tcp_xyz(grasp_pose, abort_on_contact=False)
            self.gripper.move(0.0)
            self.advance_sim(10)
            if self.check_success(self.gripper):
                dis_from_hand = self.gripper.get_distance_from_hand()
                self.gripper.move_tcp_xyz(retreat_pose, abort_on_contact=False)
                shake_label = False
                if self.check_success(self.gripper):
                    shake_label = self.gripper.shake_hand(dis_from_hand)
                    #print('finish shaking')
                if self.check_success(self.gripper) and shake_label:
                    result = True, self.gripper.read()
                    if remove:
                        contacts = self.world.get_contacts(self.gripper.body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result =  False, self.gripper.max_opening_width
            else:
                result = False, self.gripper.max_opening_width

        self.world.remove_body(self.gripper.body)
        if remove:
            self.remove_and_wait()
        return result

    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz[:2] > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res


class Gripper(object):
    """Simulated Panda hand."""
    def __init__(self, world):
        self.world = world
        self.urdf_path = Path("./data_robot/urdfs/panda/hand.urdf")

        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.015 + self.finger_depth])
        self.T_tcp_body = self.T_body_tcp.inverse()

        self.pts = np.array([
            [0, 0.04, 0],
            [0, -0.04, 0],
            [0, 0.04, -0.05],
            [0, -0.04, -0.05],
            [0, 0.0, -0.05],
        ])

    def reset(self, T_world_tcp, opening_width=None):
        opening_width = opening_width or self.max_opening_width
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)

        pybullet.changeVisualShape(self.body.uid, -1, rgbaColor=[1, 1, 1, 0.4])
        pybullet.changeVisualShape(self.body.uid, 0, rgbaColor=[1, 1, 1, 0.4])
        pybullet.changeVisualShape(self.body.uid, 1, rgbaColor=[1, 1, 1, 0.4])

        pybullet.changeDynamics(self.body.uid, 0, lateralFriction=0.75, spinningFriction=0.05)
        pybullet.changeDynamics(self.body.uid, 1, lateralFriction=0.75, spinningFriction=0.05)
        self.body.set_pose(T_world_body)

        # gripper_pts = T_world_tcp.transform_point(self.pts)
        # self.world.p.addUserDebugPoints(gripper_pts, len(gripper_pts)*[(1, 0.2, 0.1),], 10)

        # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)
        # constraint to keep fingers centered
        self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=30)

        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * opening_width, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        return len(self.world.get_contacts(self.body)) > 0

    def grasp_object_id(self):
        contacts = self.world.get_contacts(self.body)
        for contact in contacts:
            # contact = contacts[0]
            # get rid body
            grased_id = contact.bodyB
            if grased_id.uid!=self.body.uid:
                return grased_id.uid

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width

    def move_tcp_pose(self, target, eef_step1=0.002, vel1=0.10, abs=False):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp
        pos_diff = target.translation - T_world_tcp.translation

        n_steps = max(int(np.linalg.norm(pos_diff) / eef_step1), 10)
        dist_step = pos_diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel1

        key_rots = np.stack((T_world_body.rotation.as_quat(), target.rotation.as_quat()),axis=0)
        key_rots = Rotation.from_quat(key_rots)
        slerp = Slerp([0.0, 1.0], key_rots)
        times = np.linspace(0, 1, n_steps)
        orientations = slerp(times).as_quat()
        for ii in range(n_steps):
            T_world_tcp.translation += dist_step
            T_world_tcp.rotation = Rotation.from_quat(orientations[ii])
            if abs is True:
                # todo by haojie add the relation transformation later
                self.constraint.change(
                    jointChildPivot=T_world_tcp.translation,
                    jointChildFrameOrientation=T_world_tcp.rotation.as_quat(),
                    maxForce=300,
                )
            else:
                self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()

    def get_distance_from_hand(self,):
        object_id = self.grasp_object_id()
        pos, _ = pybullet.getBasePositionAndOrientation(object_id)
        dist_from_hand = np.linalg.norm(np.array(pos) - np.array(self.body.get_pose().translation))
        return dist_from_hand

    def is_dropped(self,object_id,prev_dist):
        pos,_ = pybullet.getBasePositionAndOrientation(object_id)
        dist_from_hand = np.linalg.norm(np.array(pos) - np.array(self.body.get_pose().translation))
        if np.isclose(prev_dist, dist_from_hand, atol=0.1):
            return False
        else:
            return True

    def shake_hand(self, pre_dist):
        grasp_id = self.grasp_object_id()
        current_pose = self.body.get_pose()

        a = Transform(Rotation.from_euler('y', np.pi/6), np.zeros(3))
        b = Transform(Rotation.identity(), [0, 0, 0.12])
        shake_pose =  b.inverse() * a * b

        for _ in range(3):
            self.move_tcp_pose(target=current_pose * shake_pose, eef_step1=0.01, vel1=0.3)
            if self.is_dropped(grasp_id,pre_dist):
                return False
            self.move_tcp_pose(target=current_pose * shake_pose.inverse(), eef_step1=0.01, vel1=0.3)
            if self.is_dropped(grasp_id,pre_dist):
                return False
        return True
