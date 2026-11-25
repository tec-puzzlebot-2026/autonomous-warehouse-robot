"""Manual low-level controls and independent resets, shared by the ROS and CPU runners."""
import math

import numpy as np
import omni.physics.tensors
from omni.physx import get_physx_interface, get_physx_scene_query_interface, get_physx_simulation_interface
from pxr import Gf, PhysicsSchemaTools, UsdUtils

DT = 1 / 120
RADIUS = 0.05
TRACK = 0.19
MAX_LIFT = 0.254
MIN_COMMAND = 0.012
LIFT_RATE = 0.065
WHEELS = ("wheel_left_joint", "wheel_right_joint")
SLIDES = ("forklift_mast_slide_joint", "forklift_fork_slide_joint")
DRUM_RATIOS = {"forklift_winch_joint": 1.0, "forklift_fixed_pulley_joint": -1.037037,
               "forklift_moving_pulley_joint": -0.4375}


def quaternion(row):
    return Gf.Quatd(float(row[6]), Gf.Vec3d(*(float(v) for v in row[3:6])))


def rotation(row):
    return Gf.Rotation(quaternion(row))


def yaw(row):
    x, y, z, w = map(float, row[3:7])
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


class Physics:
    def __init__(self, stage):
        self.stage = stage
        measured = stage.GetPrimAtPath("/World").GetAttribute("measuredPuzzlebotGeometry").Get()
        self.drive_track = .170 if measured else TRACK
        self.odom_track = .183 if measured else TRACK
        self.odom_radius = .0505 if measured else RADIUS
        self.sim = get_physx_simulation_interface()
        self.interface = get_physx_interface()
        # Keep the step's task dispatch bounded for this small CPU PhysX scene.
        self.interface.set_thread_count(4)
        self.query = get_physx_scene_query_interface()
        self.step_count = 0
        self.stage_id = UsdUtils.StageCache.Get().Insert(stage)
        self.sim.attach_stage(self.stage_id.ToLongInt())
        self.step()
        self.view = omni.physics.tensors.create_simulation_view("numpy", self.stage_id.ToLongInt())
        self.view.set_subspace_roots("/")
        self.robots = self.view.create_articulation_view("/World/Sets/Set_*/Robot/base_footprint")
        self.loads = self.view.create_rigid_body_view("/World/Sets/Set_*/Load")
        self.count = len(stage.GetPrimAtPath("/World/Sets").GetChildren())
        if self.robots.count != self.count or self.loads.count != self.count:
            raise RuntimeError(f"Expected {self.count} robots and loads, got {self.robots.count}, {self.loads.count}")
        self.names = self.robots.shared_metatype.dof_names
        self.wheels = [self.names.index(name) for name in WHEELS]
        self.slides = [self.names.index(name) for name in SLIDES]
        self.rotors = [self.names.index(name) for name in DRUM_RATIOS]
        self.rotor_ratios = np.array(list(DRUM_RATIOS.values()))
        self.links = {name: idx for idx, name in enumerate(self.robots.shared_metatype.link_names)}
        self.rows = {int(path.split("/")[3].split("_")[-1]): i for i, path in enumerate(self.robots.prim_paths)}
        self.load_rows = {int(path.split("/")[3].split("_")[-1]): i for i, path in enumerate(self.loads.prim_paths)}
        self.robot_keys = [PhysicsSchemaTools.sdfPathToInt(path) for path in self.robots.prim_paths]
        self.load_keys = [PhysicsSchemaTools.sdfPathToInt(path) for path in self.loads.prim_paths]
        self.indices = np.arange(self.count, dtype=np.int32)
        self.root_init = self.robots.get_root_transforms().copy()
        self.load_init = self.loads.get_transforms().copy()
        self.dof_init = self.robots.get_dof_positions().copy()
        self.position = self.dof_init.copy()
        self.velocity = np.zeros_like(self.position)
        self.desired = np.full(self.count, MIN_COMMAND)
        self.lift = self.desired.copy()
        self.twist = np.zeros((self.count, 2))
        self.last_twist = np.full(self.count, -math.inf)
        self.odom = np.zeros((self.count, 3))
        self.previous_wheels = self.position[:, self.wheels].copy()

    @property
    def time(self):
        return self.step_count * DT

    @staticmethod
    def robot_path(index):
        return f"/World/Sets/Set_{index:02d}/Robot"

    def step(self, update_usd=False):
        self.sim.simulate(DT, self.time)
        self.sim.fetch_results()
        self.step_count += 1
        if update_usd:
            self.interface.update_transformations(False, True, False, False)

    def command_drive(self, index, forward, turn):
        row = self.rows[index]
        if not (math.isfinite(forward) and math.isfinite(turn)):
            self.twist[row] = 0
            self.last_twist[row] = -math.inf
            return
        self.twist[row] = (np.clip(forward, -.22, .22), np.clip(turn, -2.0, 2.0))
        self.last_twist[row] = self.time
        if forward or turn:
            self.sim.wake_up(self.stage_id.ToLongInt(), self.robot_keys[row])

    def command_lift(self, index, height):
        if math.isfinite(height):
            row = self.rows[index]
            self.desired[row] = np.clip(height, 0, MAX_LIFT)
            self.sim.wake_up(self.stage_id.ToLongInt(), self.robot_keys[row])
            return True
        return False

    def reset(self, index):
        r, c = self.rows[index], self.load_rows[index]
        robots = np.array([r], dtype=np.int32)
        loads = np.array([c], dtype=np.int32)
        self.sim.wake_up(self.stage_id.ToLongInt(), self.robot_keys[r])
        self.sim.wake_up(self.stage_id.ToLongInt(), self.load_keys[c])
        self.robots.set_root_transforms(self.root_init, robots)
        self.robots.set_root_velocities(np.zeros((self.count, 6), dtype=np.float32), robots)
        zeros = np.zeros_like(self.position)
        self.robots.set_dof_positions(self.dof_init, robots)
        self.robots.set_dof_velocities(zeros, robots)
        self.loads.set_transforms(self.load_init, loads)
        self.loads.set_velocities(np.zeros((self.count, 6), dtype=np.float32), loads)
        self.desired[r] = self.lift[r] = MIN_COMMAND
        self.twist[r] = 0
        self.last_twist[r] = -math.inf
        self.odom[r] = 0
        self.previous_wheels[r] = self.dof_init[r, self.wheels]

    def tick(self, update_usd=False, firmware_lift=False):
        active = self.time - self.last_twist < .35
        forward, turn = (self.twist * active[:, None]).T
        self.velocity[:, self.wheels[0]] = (forward - turn * self.drive_track/2) / RADIUS
        self.velocity[:, self.wheels[1]] = (forward + turn * self.drive_track/2) / RADIUS
        if not firmware_lift:
            self.lift += np.clip(self.desired-self.lift, -DT*LIFT_RATE, DT*LIFT_RATE)
        q = self.lift / 2
        self.position[:, self.slides] = q[:, None]
        phases = self.robots.get_dof_positions()[:, self.rotors]
        error = q[:, None]*self.rotor_ratios[None, :]/.007 - phases
        # Track the drum/pulley phase modulo a full turn, using the angular velocity drive.
        self.velocity[:, self.rotors] = np.clip(12*np.arctan2(np.sin(error), np.cos(error)), -8, 8)
        self.robots.set_dof_position_targets(self.position, self.indices)
        self.robots.set_dof_velocity_targets(self.velocity, self.indices)
        self.step(update_usd)
        current_wheels = self.robots.get_dof_positions()[:, self.wheels]
        dl, dr = ((current_wheels-self.previous_wheels)*self.odom_radius).T
        advance = (dl+dr)/2
        turn = (dr-dl)/self.odom_track
        mid = self.odom[:, 2]+turn/2
        self.odom[:, 0] += advance*np.cos(mid)
        self.odom[:, 1] += advance*np.sin(mid)
        self.odom[:, 2] += turn
        self.previous_wheels[:] = current_wheels

    def link_pose(self, index, name, transforms=None):
        if transforms is None:
            transforms = self.robots.get_link_transforms()
        return transforms[self.rows[index], self.links[name]].copy()

    def range_ray(self, index, start, direction, distance, robust=False):
        # Lidar and IR origins can be inside their own proxy. Filter every hit by robot path.
        own = self.robot_path(index) + "/"
        nearest = math.inf
        def on_hit(hit):
            nonlocal nearest
            if not (str(hit.rigid_body).startswith(own) or str(hit.collision).startswith(own)):
                nearest = min(nearest, hit.distance)
            return True
        start = tuple(map(float, start))
        direction = tuple(map(float, direction))
        if not robust:
            first = self.query.raycast_closest(start, direction, distance)
            if not first["hit"]:
                return math.inf
            if not (str(first["rigidBody"]).startswith(own) or str(first["collision"]).startswith(own)):
                return float(first["distance"])
        self.query.raycast_all(start, direction, distance, on_hit)
        return float(nearest)

    def close(self):
        self.sim.detach_stage()
        UsdUtils.StageCache.Get().Erase(self.stage_id)
