from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class DogUrdfRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 49
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = False
        include_lin_vel = True
        curriculum = True
        terrain_proportions = [0.4, 0.2, 0.15, 0.15, 0.1]
        num_rows = 10
        num_cols = 10
        max_init_terrain_level = 3

    class commands(LeggedRobotCfg.commands):
        num_commands = 5
        resampling_time = 10.
        heading_command = True
        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]
            height = [0.15, 0.28]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.25]
        default_joint_angles = {
            'FL_hip_joint': 0.1, 'RL_hip_joint': 0.1, 'FR_hip_joint': 0.1, 'RR_hip_joint': 0.1,
            'FL_thigh_joint': 0.8, 'RL_thigh_joint': 1.0, 'FR_thigh_joint': 0.8, 'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5, 'RL_calf_joint': -1.5, 'FR_calf_joint': -1.5, 'RR_calf_joint': -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 20.}
        damping = {'joint': 0.5}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/dog_urdf/urdf/dog_urdf.urdf'
        name = "dog_urdf"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1
        collapse_fixed_joints = False
        armature = 0.01

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_joint_armature = True
        joint_armature_range = [0.005, 0.02]
        randomize_joint_damping = True
        joint_damping_range = [0.8, 1.2]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -5.0
            orientation = -1.0
            base_height = -1.0
            tracking_lin_vel = 1.5
            feet_air_time = 10.0
            foot_slip = -0.1
            action_rate = -0.05

    class noise(LeggedRobotCfg.noise):
        noise_level = 1.0

class DogUrdfRoughCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_dog_urdf'
        max_iterations = 10000
        save_interval = 50
