import mujoco
import numpy as np
from dm_control import mjcf

from ogbench.manipspace import lie
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv


class CubeEnv(ManipSpaceEnv):
    """Cube environment.

    This environment consists of a single or multiple cubes. The goal is to move the cubes to target positions. It
    supports the following variants:
    - `env_type`: 'single', 'double', 'triple', 'quadruple'.
    """

    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        """Initialize the Cube environment.

        Args:
            env_type: Environment type. One of 'single', 'double', 'triple', or 'quadruple'.
            permute_blocks: Whether to randomly permute the order of the blocks at task initialization.
            *args: Additional arguments to pass to the parent class.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        self._env_type = env_type
        self._permute_blocks = permute_blocks

        if self._env_type == 'single':
            self._num_cubes = 1
        elif self._env_type == 'double':
            self._num_cubes = 2
        elif self._env_type == 'triple':
            self._num_cubes = 3
        elif self._env_type == 'quadruple':
            self._num_cubes = 4
        else:
            raise ValueError(f'Invalid env_type: {env_type}')

        super().__init__(*args, **kwargs)

        # Define constants.
        self._cube_colors = np.array(
            [
                self._colors['red'],
                self._colors['blue'],
                self._colors['orange'],
                self._colors['green'],
            ]
        )
        self._cube_success_colors = np.array(
            [
                self._colors['lightred'],
                self._colors['lightblue'],
                self._colors['lightorange'],
                self._colors['lightgreen'],
            ]
        )

        # Target info.
        self._target_task = 'cube'
        # The target cube position is stored in the mocap object.
        self._target_block = 0

    def set_tasks(self):
        if self._env_type == 'single':
            self.task_infos = [
                dict(
                    task_name='task1_horizontal',
                    init_xyzs=np.array([[0.425, 0.1, 0.02]]),
                    goal_xyzs=np.array([[0.425, -0.1, 0.02]]),
                ),
                dict(
                    task_name='task2_vertical1',
                    init_xyzs=np.array([[0.35, 0.0, 0.02]]),
                    goal_xyzs=np.array([[0.50, 0.0, 0.02]]),
                ),
                dict(
                    task_name='task3_vertical2',
                    init_xyzs=np.array([[0.50, 0.0, 0.02]]),
                    goal_xyzs=np.array([[0.35, 0.0, 0.02]]),
                ),
                dict(
                    task_name='task4_diagonal1',
                    init_xyzs=np.array([[0.35, -0.2, 0.02]]),
                    goal_xyzs=np.array([[0.50, 0.2, 0.02]]),
                ),
                dict(
                    task_name='task5_diagonal2',
                    init_xyzs=np.array([[0.35, 0.2, 0.02]]),
                    goal_xyzs=np.array([[0.50, -0.2, 0.02]]),
                ),
            ]
        elif self._env_type == 'double':
            self.task_infos = [
                dict(
                    task_name='task1_single_pnp',
                    init_xyzs=np.array(
                        [
                            [0.425, 0.0, 0.02],
                            [0.425, -0.1, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, 0.0, 0.02],
                            [0.425, 0.1, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task2_double_pnp1',
                    init_xyzs=np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.50, -0.1, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.35, 0.1, 0.02],
                            [0.50, 0.1, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task3_double_pnp2',
                    init_xyzs=np.array(
                        [
                            [0.35, 0.0, 0.02],
                            [0.50, 0.0, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, -0.2, 0.02],
                            [0.425, 0.2, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task4_swap',
                    init_xyzs=np.array(
                        [
                            [0.425, -0.1, 0.02],
                            [0.425, 0.1, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, 0.1, 0.02],
                            [0.425, -0.1, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task5_stack',
                    init_xyzs=np.array(
                        [
                            [0.425, -0.2, 0.02],
                            [0.425, 0.2, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, 0.0, 0.02],
                            [0.425, 0.0, 0.06],
                        ]
                    ),
                ),
            ]
        elif self._env_type == 'triple':
            self.task_infos = [
                dict(
                    task_name='task1_single_pnp',
                    init_xyzs=np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.35, 0.1, 0.02],
                            [0.50, -0.1, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.35, 0.1, 0.02],
                            [0.50, 0.1, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task2_triple_pnp',
                    init_xyzs=np.array(
                        [
                            [0.35, -0.2, 0.02],
                            [0.35, 0.0, 0.02],
                            [0.35, 0.2, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.50, 0.0, 0.02],
                            [0.50, 0.2, 0.02],
                            [0.50, -0.2, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task3_pnp_from_stack',
                    init_xyzs=np.array(
                        [
                            [0.425, 0.2, 0.02],
                            [0.425, 0.2, 0.06],
                            [0.425, 0.2, 0.10],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.50, -0.2, 0.02],
                            [0.50, 0.0, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task4_cycle',
                    init_xyzs=np.array(
                        [
                            [0.35, 0.0, 0.02],
                            [0.50, -0.1, 0.02],
                            [0.50, 0.1, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.50, -0.1, 0.02],
                            [0.50, 0.1, 0.02],
                            [0.35, 0.0, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task5_stack',
                    init_xyzs=np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.50, -0.2, 0.02],
                            [0.50, 0.0, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, 0.2, 0.02],
                            [0.425, 0.2, 0.06],
                            [0.425, 0.2, 0.10],
                        ]
                    ),
                ),
            ]
        elif self._env_type == 'quadruple':
            self.task_infos = [
                dict(
                    task_name='task1_double_pnp',
                    init_xyzs=np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.35, 0.1, 0.02],
                            [0.50, -0.1, 0.02],
                            [0.50, 0.1, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.35, -0.25, 0.02],
                            [0.35, 0.1, 0.02],
                            [0.50, -0.1, 0.02],
                            [0.50, 0.25, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task2_quadruple_pnp',
                    init_xyzs=np.array(
                        [
                            [0.325, -0.2, 0.02],
                            [0.325, 0.2, 0.02],
                            [0.525, -0.2, 0.02],
                            [0.525, 0.2, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.375, 0.1, 0.02],
                            [0.475, 0.1, 0.02],
                            [0.375, -0.1, 0.02],
                            [0.475, -0.1, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task3_pnp_from_square',
                    init_xyzs=np.array(
                        [
                            [0.425, -0.02, 0.02],
                            [0.425, 0.02, 0.02],
                            [0.425, -0.02, 0.06],
                            [0.425, 0.02, 0.06],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.525, -0.2, 0.02],
                            [0.325, 0.2, 0.02],
                            [0.325, -0.2, 0.02],
                            [0.525, 0.2, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task4_cycle',
                    init_xyzs=np.array(
                        [
                            [0.525, -0.1, 0.02],
                            [0.525, 0.1, 0.02],
                            [0.325, 0.1, 0.02],
                            [0.325, -0.1, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.525, 0.1, 0.02],
                            [0.325, 0.1, 0.02],
                            [0.325, -0.1, 0.02],
                            [0.525, -0.1, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task5_stack',
                    init_xyzs=np.array(
                        [
                            [0.50, -0.05, 0.02],
                            [0.50, -0.2, 0.02],
                            [0.35, -0.2, 0.02],
                            [0.35, -0.05, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, 0.2, 0.02],
                            [0.425, 0.2, 0.06],
                            [0.425, 0.2, 0.10],
                            [0.425, 0.2, 0.14],
                        ]
                    ),
                ),
            ]

        if self._reward_task_id == 0:
            self._reward_task_id = 2  # Default task.

    def add_objects(self, arena_mjcf):
        # Add cube scene.
        cube_outer_mjcf = mjcf.from_path((self._desc_dir / 'cube_outer.xml').as_posix())
        arena_mjcf.include_copy(cube_outer_mjcf)

        # Add `num_cubes` cubes to the scene.
        distance = 0.05
        for i in range(self._num_cubes):
            cube_mjcf = mjcf.from_path((self._desc_dir / 'cube_inner.xml').as_posix())
            pos = -distance * (self._num_cubes - 1) + 2 * distance * i
            cube_mjcf.find('body', 'object_0').pos[1] = pos
            cube_mjcf.find('body', 'object_target_0').pos[1] = pos
            for tag in ['body', 'joint', 'geom', 'site']:
                for item in cube_mjcf.find_all(tag):
                    if hasattr(item, 'name') and item.name is not None and item.name.endswith('_0'):
                        item.name = item.name[:-2] + f'_{i}'
            arena_mjcf.include_copy(cube_mjcf)

        # Save cube geoms.
        self._cube_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_geoms_list.append(arena_mjcf.find('body', f'object_{i}').find_all('geom'))
        self._cube_target_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_target_geoms_list.append(arena_mjcf.find('body', f'object_target_{i}').find_all('geom'))

        # Add cameras.
        cameras = {
            'front': {
                'pos': (1.287, 0.000, 0.509),
                'xyaxes': (0.000, 1.000, 0.000, -0.342, 0.000, 0.940),
            },
            'front_pixels': {
                'pos': (1.053, -0.014, 0.639),
                'xyaxes': (0.000, 1.000, 0.000, -0.628, 0.001, 0.778),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

    def post_compilation_objects(self):
        # Cube geom IDs.
        self._cube_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in cube_geoms] for cube_geoms in self._cube_geoms_list
        ]
        self._cube_target_mocap_ids = [
            self._model.body(f'object_target_{i}').mocapid[0] for i in range(self._num_cubes)
        ]
        self._cube_target_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in cube_target_geoms]
            for cube_target_geoms in self._cube_target_geoms_list
        ]

    def initialize_episode(self):
        # Set cube colors.
        for i in range(self._num_cubes):
            for gid in self._cube_geom_ids_list[i]:
                self._model.geom(gid).rgba = self._cube_colors[i]
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        if self._mode == 'data_collection':
            # Randomize the scene.

            self.initialize_arm()

            # Randomize object positions and orientations.
            for i in range(self._num_cubes):
                xy = self.np_random.uniform(*self._object_sampling_bounds)
                obj_pos = (*xy, 0.02)
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori

            # Set a new target.
            self.set_new_target(return_info=False)
        else:
            # Set object positions and orientations based on the current task.

            if self._permute_blocks:
                # Randomize the order of the cubes when there are multiple cubes.
                permutation = self.np_random.permutation(self._num_cubes)
            else:
                permutation = np.arange(self._num_cubes)
            init_xyzs = self.cur_task_info['init_xyzs'].copy()[permutation]
            goal_xyzs = self.cur_task_info['goal_xyzs'].copy()[permutation]

            # First, force set the current scene to the goal state to obtain the goal observation.
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            self.initialize_arm()
            for i in range(self._num_cubes):
                self._data.joint(f'object_joint_{i}').qpos[:3] = goal_xyzs[i]
                self._data.joint(f'object_joint_{i}').qpos[3:] = lie.SO3.identity().wxyz.tolist()
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
            mujoco.mj_forward(self._model, self._data)

            # Do a few random steps to make the scene stable.
            for _ in range(2):
                self.step(self.action_space.sample())

            # Save the goal observation.
            self._cur_goal_ob = (
                self.compute_oracle_observation() if self._use_oracle_rep else self.compute_observation()
            )
            if self._render_goal:
                self._cur_goal_rendered = self.render()
            else:
                self._cur_goal_rendered = None

            # Now, do the actual reset.
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            for i in range(self._num_cubes):
                # Randomize the position and orientation of the cube slightly.
                obj_pos = init_xyzs[i].copy()
                obj_pos[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()

        # Forward kinematics to update site positions.
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def set_new_target(self, return_info=True, p_stack=0.5):
        """Set a new random target for data collection.

        Args:
            return_info: Whether to return the observation and reset info.
            p_stack: Probability of stacking the target block on top of another block when there are multiple blocks.
        """
        assert self._mode == 'data_collection'

        block_xyzs = np.array([self._data.joint(f'object_joint_{i}').qpos[:3] for i in range(self._num_cubes)])

        # Compute the top blocks.
        top_blocks = []
        for i in range(self._num_cubes):
            for j in range(self._num_cubes):
                if i == j:
                    continue
                if block_xyzs[j][2] > block_xyzs[i][2] and np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2]) < 0.02:
                    break
            else:
                top_blocks.append(i)

        # Pick one of the top cubes as the target.
        self._target_block = self.np_random.choice(top_blocks)

        stack = len(top_blocks) >= 2 and self.np_random.uniform() < p_stack
        if stack:
            # Stack the target block on top of another block.
            block_idx = self.np_random.choice(list(set(top_blocks) - {self._target_block}))
            block_pos = self._data.joint(f'object_joint_{block_idx}').qpos[:3]
            tar_pos = np.array([block_pos[0], block_pos[1], block_pos[2] + 0.04])
        else:
            # Randomize target position.
            xy = self.np_random.uniform(*self._target_sampling_bounds)
            tar_pos = (*xy, 0.02)
        # Randomize target orientation.
        yaw = self.np_random.uniform(0, 2 * np.pi)
        tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()

        # Only show the target block.
        for i in range(self._num_cubes):
            if i == self._target_block:
                # Set the target position and orientation.
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = tar_pos
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = tar_ori
            else:
                # Move the non-target blocks out of the way.
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = (0, 0, -0.3)
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()

        # Set the target colors.
        for i in range(self._num_cubes):
            if self._visualize_info and i == self._target_block:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def _compute_successes(self):
        """Compute object successes."""
        cube_successes = []
        for i in range(self._num_cubes):
            obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = self._data.mocap_pos[self._cube_target_mocap_ids[i]]
            if np.linalg.norm(obj_pos - tar_pos) <= 0.04:
                cube_successes.append(True)
            else:
                cube_successes.append(False)

        return cube_successes

    def post_step(self):
        # Check if the cubes are in the target positions.
        cube_successes = self._compute_successes()
        if self._mode == 'data_collection':
            self._success = cube_successes[self._target_block]
        else:
            self._success = all(cube_successes)

        # Adjust the colors of the cubes based on success.
        for i in range(self._num_cubes):
            if self._visualize_info and (self._mode == 'task' or i == self._target_block):
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

            if self._visualize_info and cube_successes[i]:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_success_colors[i, :3]
            else:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]

    def add_object_info(self, ob_info):
        # Cube positions and orientations.
        for i in range(self._num_cubes):
            ob_info[f'privileged/block_{i}_pos'] = self._data.joint(f'object_joint_{i}').qpos[:3].copy()
            ob_info[f'privileged/block_{i}_quat'] = self._data.joint(f'object_joint_{i}').qpos[3:].copy()
            ob_info[f'privileged/block_{i}_yaw'] = np.array(
                [lie.SO3(wxyz=self._data.joint(f'object_joint_{i}').qpos[3:]).compute_yaw_radians()]
            )

        if self._mode == 'data_collection':
            # Target cube info.
            ob_info['privileged/target_task'] = self._target_task

            target_mocap_id = self._cube_target_mocap_ids[self._target_block]
            ob_info['privileged/target_block'] = self._target_block
            ob_info['privileged/target_block_pos'] = self._data.mocap_pos[target_mocap_id].copy()
            ob_info['privileged/target_block_yaw'] = np.array(
                [lie.SO3(wxyz=self._data.mocap_quat[target_mocap_id]).compute_yaw_radians()]
            )
            
        for i in range(self._num_cubes):
            obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = self._data.mocap_pos[self._cube_target_mocap_ids[i]]
            init_pos = self.cur_task_info["init_xyzs"][i]
            ob_info[f"distance_from_goal_{i}"] = np.linalg.norm(obj_pos - tar_pos)
            ob_info[f"distance_from_start_{i}"] = np.linalg.norm(obj_pos - init_pos)
            
        return ob_info

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0

            ob_info = self.compute_ob_info()
            ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
            ]
            for i in range(self._num_cubes):
                ob.extend(
                    [
                        (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                        ob_info[f'privileged/block_{i}_quat'],
                        np.cos(ob_info[f'privileged/block_{i}_yaw']),
                        np.sin(ob_info[f'privileged/block_{i}_yaw']),
                    ]
                )

            return np.concatenate(ob)

    def compute_oracle_observation(self):
        """Return the oracle goal representation of the current state."""
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0

        ob_info = self.compute_ob_info()
        ob = []
        for i in range(self._num_cubes):
            ob.append((ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler)

        return np.concatenate(ob)

    def compute_reward(self, ob, action):
        if self._reward_task_id is None:
            return super().compute_reward(ob, action)

        # Compute the reward based on the task.
        successes = self._compute_successes()
        reward = float(sum(successes) - len(successes))
        return reward
