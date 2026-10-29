"""Reusable MuJoCo raycasting, independent of any learning environment."""

import mujoco
import numpy as np


class Raycaster:
    def __init__(self, model, data, robot_indicator_mujoco_body_ids):
        """Map each robot body ID to the body defining its sensor pose."""
        self.model = model
        self.data = data
        self.robot_indicator_mujoco_body_ids = dict(robot_indicator_mujoco_body_ids)
        self.raycast_rot_mats = {}
        self.raycast_ray_dirs_buf = {}
        self.raycast_geomids_buf = {}
        self.raycast_distances_buf = {}
        self.raycast_origins_buf = {}
        # Include all six geometry groups, including static geometry.
        self._raycast_geomgroup = np.ones(6, dtype=np.uint8)

    def setup_raycast(self, robot_mujoco_body_id, nrays, angle_covered_degrees):
        """Precompute directions and buffers once for each robot."""
        if isinstance(nrays, bool) or not isinstance(nrays, (int, np.integer)) or nrays < 1:
            raise ValueError("nrays must be a positive integer")
        if not np.isfinite(angle_covered_degrees) or not 0 <= angle_covered_degrees <= 360:
            raise ValueError("angle_covered_degrees must be between 0 and 360")
        if not 0 <= robot_mujoco_body_id < self.model.nbody:
            raise ValueError("Invalid robot body ID")
        indicator_id = self.robot_indicator_mujoco_body_ids[robot_mujoco_body_id]
        if not 0 <= indicator_id < self.model.nbody:
            raise ValueError("Invalid indicator body ID")

        # Spread rays symmetrically; a single ray always points forward.
        angles = np.linspace(-angle_covered_degrees / 2, angle_covered_degrees / 2, nrays)
        if nrays == 1:
            angles[0] = 0
        rot_mats = np.zeros((nrays, 2, 2), dtype=np.float64)
        for i, angle in enumerate(np.deg2rad(angles)):
            rot_mats[i] = [[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]]

        self.raycast_rot_mats[robot_mujoco_body_id] = rot_mats
        self.raycast_ray_dirs_buf[robot_mujoco_body_id] = np.zeros((nrays, 3), dtype=np.float64)
        self.raycast_geomids_buf[robot_mujoco_body_id] = np.full(nrays, -1, dtype=np.int32)
        self.raycast_distances_buf[robot_mujoco_body_id] = np.full(nrays, -1.0, dtype=np.float64)
        self.raycast_origins_buf[robot_mujoco_body_id] = np.zeros(3, dtype=np.float64)

    def perform_raycast(self, robot_mujoco_body_id, ray_length=10):
        """Return (geometry IDs, distances), using -1 for missing hits.

        Call mj_forward or mj_step after changing the simulation state.
        Returned arrays are reused by the next call for the same robot;
        copy them if you need to retain a previous reading.
        """
        if not np.isfinite(ray_length) or ray_length <= 0:
            raise ValueError("ray_length must be finite and positive")
        if robot_mujoco_body_id not in self.raycast_rot_mats:
            raise ValueError("Call setup_raycast for this robot first")

        rot_mats = self.raycast_rot_mats[robot_mujoco_body_id]
        ray_dirs = self.raycast_ray_dirs_buf[robot_mujoco_body_id]
        geomids = self.raycast_geomids_buf[robot_mujoco_body_id]
        distances = self.raycast_distances_buf[robot_mujoco_body_id]
        #Position of the robot
        ray_start = self.raycast_origins_buf[robot_mujoco_body_id]
        nrays = len(geomids)
        geomids.fill(-1)
        distances.fill(-1.0)

        indicator_id = self.robot_indicator_mujoco_body_ids[robot_mujoco_body_id]
        # where the robot is 
        indicator_mat = self.data.xmat[indicator_id].reshape(3, 3)
        # where the robot is looking
        world_forward = indicator_mat[:, 0]  
        np.multiply(world_forward, 1e-6, out=ray_start) #little offset 

        #the rays start from the position of the robot  
        ray_start += self.data.xpos[indicator_id]

        # Preserve the supplied world-Z rotation, including the forward Z component.
        np.matmul(rot_mats, world_forward[:2], out=ray_dirs[:, :2])
        ray_dirs[:, 2] = world_forward[2]
        # Unit directions make the returned values distances, not scaled fractions.
        # Each robot needs its own call because mj_multiRay uses a common origin.
        mujoco.mj_multiRay(
            self.model, self.data, ray_start, ray_dirs.reshape(-1),
            self._raycast_geomgroup, True, -1, geomids, distances, None,
            nrays, ray_length,
        )

        # The cutoff culls distant geoms; enforce the exact ray segment as well.
        for i in range(nrays):
            if distances[i] > ray_length:
                geomids[i] = -1
                distances[i] = -1.0
        return geomids, distances
