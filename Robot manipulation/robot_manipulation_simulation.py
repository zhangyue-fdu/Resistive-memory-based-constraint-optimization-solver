import pybullet as p
import pybullet_data
import numpy as np
from scipy.optimize import minimize
import time
import os
import pandas as pd

# ===== Initialize PyBullet simulation environment =====
if p.isConnected() == 0:
    p.connect(p.GUI)  # connect to the physics engine (GUI mode)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # add default data path
p.setGravity(0, 0, -9.8)  # set gravity

# Configure PyBullet window
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # hide GUI overlays
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,  # camera-scene distance
    cameraYaw=-50,        # horizontal angle
    cameraPitch=-35,     # vertical angle
    cameraTargetPosition=[-0.5, -0.5, 0.2]  # camera focus point
)

# Resize the GUI window
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # enable rendering

# ===== Load scene and robot arm =====
plane_id = p.loadURDF("plane.urdf")  # Load the ground
panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)  # Load the robotic arm
# Print joint / link info
for i in range(p.getNumJoints(panda_id)):
    joint_info = p.getJointInfo(panda_id, i)
    print(
        f"Joint Index: {i}, "
        f"Name: {joint_info[1].decode('utf-8')}, "
        f"Type: {joint_info[2]}, "
        f"Link Name: {joint_info[12].decode('utf-8')}"
    )
# Load object and cup
object_id = p.loadURDF("cube_small.urdf", basePosition=[-0.8, 0, 0.02], globalScaling=0.5)  # Objects
cup_path = "cup.urdf"
cup_position = [-0.8, -0.3, 0.02]
cup_id = p.loadURDF(cup_path, basePosition=cup_position)

# Target pose inside the cup
target_position = [-0.9, -0.32, 0.2]  # Placement point inside the cup

# ===== Parameters =====
NUM_JOINTS = 6  # Number of joints of the robotic arm
END_EFFECTOR_INDEX = 7  # Link index of the end effector
T = 10  # time steps for optimization

# Starting position and grab position
start_position = [-0.4, 0, 0.2]  # initial pose
pick_position = [-0.8, 0, 0.03]  # pick pose

# Joint limits
joint_limits = [
    (-2.8973, 2.8973),  # Joint 1
    (-1.7628, 1.7628),  # Joint 2
    (-2.8973, 2.8973),  # Joint 3
    (-3.0718, -0.0698), # Joint 4
    (-2.8973, 2.8973),  # Joint 5
    (-0.0175, 3.7525),  # Joint 6
    (-2.8973, 2.8973),  # Joint 7
]

# ===== Utility functions =====
def clamp_joint_angles(angles):
    """
    Clamp joint angles to within the allowed range.
    :param angles: Joint angles to check
    :return: Limited joint angles
    """
    clamped_angles = []
    for i, angle in enumerate(angles):
        lower, upper = joint_limits[i]
        clamped_angles.append(np.clip(angle, lower, upper))
    return np.array(clamped_angles)

def control_gripper(close=True):
    """
    Controls the opening and closing of the gripper.
    :param close: Whether to close the gripper (True for closed, False for open)
    """
    target_position = 0.0 if close else 0.04  # Target position of the gripper when closing
    for joint in [9, 10]:  # Joint index of the gripper
        p.setJointMotorControl2(
            panda_id,
            joint,
            p.POSITION_CONTROL,
            targetPosition=target_position,
            force=10  # Set the gripper force
        )
    # Wait for the gripper action to complete
    for _ in range(50):
        p.stepSimulation()
        time.sleep(0.01)

def attach_object_to_gripper(object_id):
    """
    Attach the object to the gripper of the robot arm.
    :param object_id: The ID of the object
    """
    constraint_id = p.createConstraint(
        parentBodyUniqueId=panda_id,
        parentLinkIndex=END_EFFECTOR_INDEX,
        childBodyUniqueId=object_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    return constraint_id

def detach_object_from_gripper(constraint_id):
    """
    Separate the object from the gripper.
    :param constraint_id: The constraint ID
    """
    p.removeConstraint(constraint_id)

# ===== Optimization problem =====
def objective(x):
    """
    Objective function: Minimize path smoothness and control energy consumption.
    :param x: Optimization variable (joint angle sequence)
    :return: Target value
    """
    x = x.reshape(-1, NUM_JOINTS)
    smoothness_cost = np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1)**2)
    control_cost = np.sum(np.linalg.norm(x, axis=1)**2)
    return 10*(smoothness_cost + 0.1 * control_cost)  #Balance path smoothness and control energy consumption

def start_end_constraint(x, start_angles, end_angles):
    """
    Start and end point constraints.
    :param x: Optimization variable (joint angle sequence)
    :param start_angles: Starting joint angles
    :param end_angles: Target joint angles
    :return: Constraint value
    """
    x = x.reshape(-1, NUM_JOINTS)
    start_error = np.linalg.norm(x[0] - start_angles)
    end_error = np.linalg.norm(x[-1] - end_angles)
    tolerance = 1e-3  # Allow a certain range of error
    return np.array([start_error - tolerance, end_error - tolerance])


def plan_path(start_angles, end_angles):
    """
    Use the optimizer to plan a path and save the results of each iteration to an Excel file.
    :param start_angles: Starting joint angles
    :param end_angles: Ending joint angles
    :return: Optimized path
    """
    # Assume the following constants have been defined and adjust them according to the actual situation
    T = 10  # Time step example
    NUM_JOINTS = len(start_angles)  # Number of joints

    # 初始化历史记录
    history = []

    def callback(xk):
        """Record the parameters, objective function value and constraint violation degree of each iteration"""
        obj_value = objective(xk)
        constraint_val = start_end_constraint(xk, start_angles, end_angles)
        constraint_violation = np.linalg.norm(constraint_val)
        iteration = len(history)
        history.append({
            'iteration': iteration,
            'objective': obj_value,
            'constraint_violation': constraint_violation,
            'parameters': xk.copy()
        })

    initial_guess = np.linspace(start_angles, end_angles, num=T).flatten()
    constraints = [
        {
            "type": "eq",
            "fun": start_end_constraint,
            "args": (start_angles, end_angles),
        }
    ]
    options = {"maxiter": 5000, "ftol": 1e-6}

    # Perform optimization and add callback function
    result = minimize(
        objective,
        initial_guess,
        constraints=constraints,
        method="SLSQP",
        options=options,
        callback=callback
    )

    # Save history to Excel
    if result.success:
        # Convert to DataFrame and expand parameters
        df = pd.DataFrame(history)
        params_df = df['parameters'].apply(pd.Series)
        params_df.columns = [f'param_{i}' for i in params_df.columns]
        df = pd.concat([df.drop('parameters', axis=1), params_df], axis=1)
        df.to_excel('optimization_history.xlsx', index=False)

        return result.x.reshape(T, NUM_JOINTS)
    else:
        raise RuntimeError("Path planning failed: " + result.message)

def plan_path_1(start_angles, end_angles):
    """
    Use the optimizer to plan a path and save the results of each iteration to an Excel file.
    :param start_angles: Starting joint angles
    :param end_angles: Ending joint angles
    :return: Optimized path
    """
    # Assume the following constants have been defined and adjust them according to the actual situation
    T = 10  # Time step example
    NUM_JOINTS = len(start_angles)  # Number of joints

    # Initialize history
    history = []

    def callback(xk):
        """Record the parameters, objective function value and constraint violation degree of each iteration"""
        obj_value = objective(xk)
        constraint_val = start_end_constraint(xk, start_angles, end_angles)
        constraint_violation = np.linalg.norm(constraint_val)
        iteration = len(history)
        history.append({
            'iteration': iteration,
            'objective': obj_value,
            'constraint_violation': constraint_violation,
            'parameters': xk.copy()
        })

    initial_guess = np.linspace(start_angles, end_angles, num=T).flatten()
    constraints = [
        {
            "type": "eq",
            "fun": start_end_constraint,
            "args": (start_angles, end_angles),
        }
    ]
    options = {"maxiter": 5000, "ftol": 1e-6}

    # Perform optimization and add callback function
    result = minimize(
        objective,
        initial_guess,
        constraints=constraints,
        method="SLSQP",
        options=options,
        callback=callback
    )

    # Save history to Excel
    if result.success:
        # Convert to DataFrame and expand parameters
        df = pd.DataFrame(history)
        params_df = df['parameters'].apply(pd.Series)
        params_df.columns = [f'param_{i}' for i in params_df.columns]
        df = pd.concat([df.drop('parameters', axis=1), params_df], axis=1)
        df.to_excel('optimization_history_1.xlsx', index=False)

        return result.x.reshape(T, NUM_JOINTS)
    else:
        raise RuntimeError("Path planning failed: " + result.message)


# ===== Initialize joint angles (key modification points) =====
# Define the gripping posture: Rotate 180 degrees around the Y axis, pointing the gripper's Z axis downward.
target_pick_orientation = p.getQuaternionFromEuler([0, np.pi, 0])

start_angles = clamp_joint_angles(
    np.array(p.calculateInverseKinematics(panda_id, END_EFFECTOR_INDEX, start_position,targetOrientation=target_pick_orientation))[:NUM_JOINTS]
)

# Calculate the joint angles of the grasping position (add posture parameters)
pick_angles = clamp_joint_angles(
    np.array(p.calculateInverseKinematics(
        panda_id,
        END_EFFECTOR_INDEX,
        pick_position,
        targetOrientation=target_pick_orientation  # Add new posture parameters
    ))[:NUM_JOINTS]
)

# Calculate joint angles for placement (using the same pose)
place_angles = clamp_joint_angles(
    np.array(p.calculateInverseKinematics(
        panda_id,
        END_EFFECTOR_INDEX,
        target_position,
        targetOrientation=target_pick_orientation  # Maintain consistent posture
    ))[:NUM_JOINTS]
)


# ===== Add coordinate system drawing function =====
def draw_coordinate_system(position, orientation, line_length=0.2, line_width=2, text=""):
    """
    Draws a coordinate system (XYZ axes) at the specified position and orientation.
    :param position: Coordinate system origin position [x,y,z]
    :param orientation: Quaternion orientation [x,y,z,w]
    :param line_length: Coordinate axis length (m)
    :param line_width: Line width
    :param text: Text label to display
    """
    # Convert quaternion to rotation matrix
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

    # Calculate the direction vector of each axis
    x_axis = position + rotation_matrix[:, 0] * line_length
    y_axis = position + rotation_matrix[:, 1] * line_length
    z_axis = position + rotation_matrix[:, 2] * line_length

    # Draw the axes
    p.addUserDebugLine(position, x_axis, [1, 0, 0], lineWidth=line_width, lifeTime=0.1)  # X axis red
    p.addUserDebugLine(position, y_axis, [0, 1, 0], lineWidth=line_width, lifeTime=0.1)  # Y axis green
    p.addUserDebugLine(position, z_axis, [0, 0, 1], lineWidth=line_width, lifeTime=0.1)  # Z axis blue

    # Add text labels
    p.addUserDebugText(text, position, textColorRGB=[1, 1, 0], textSize=1.2, lifeTime=0.1)


# ===== Add base coordinate system drawing after loading the robot =====
# Add after the code that loads the robot:
# panda_id = p.loadURDF(...)

# Get the base position and attitude
base_pos, base_orient = p.getBasePositionAndOrientation(panda_id)
draw_coordinate_system(base_pos, base_orient, line_length=0.3, line_width=3, text="Base")
# ===== Add target coordinate system drawing after loading the target position =====
# Add after defining target_position:
draw_coordinate_system(target_position, [0,0,0,1], line_length=0.2, text="Target")

# ===== Execution Path =====
def execute_path(path):
    """
    Execute the planned path.
    :param path: path (sequence of joint angles)
    """
    for angles in path:
        for joint_index in range(NUM_JOINTS):
            p.setJointMotorControl2(
                panda_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=angles[joint_index],
            )
        p.stepSimulation()
        # Get the end effector coordinates
        end_effector_state = p.getLinkState(panda_id, END_EFFECTOR_INDEX)
        end_effector_pos = end_effector_state[0]  # The position coordinate is the first element of the tuple
        print(f"Time step: end coordinate = {end_effector_pos}, posture coordinate = {end_effector_state[1]}")
        # Get the end effector status
        end_effector_state = p.getLinkState(panda_id, END_EFFECTOR_INDEX)
        end_pos = end_effector_state[0]
        end_orient = end_effector_state[1]
        # Draw the end coordinate system
        draw_coordinate_system(end_pos, end_orient, line_length=0.15, text="End Effector")
        time.sleep(0.05)

# ===== Restore the robot arm to its initial state =====
def reset_arm_to_start():
    """
    Return the robot arm to its initial position.
    """
    print("Restore the robot arm to its initial position...")
    path_to_start = plan_path(place_angles, start_angles)
    execute_path(path_to_start)


# ===== Add video recording =====
video_path = "simulation_video.mp4"
print(f"The video will be saved to:{os.path.abspath(video_path)}")
log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)


def visualize_end_effector():
    # Get the end effector status
    end_effector_state = p.getLinkState(panda_id, END_EFFECTOR_INDEX)
    end_pos = end_effector_state[0]
    end_orient = end_effector_state[1]

    # Draw the coordinate system
    draw_coordinate_system(end_pos, end_orient, line_length=0.2, line_width=3, text="End Effector")

# ===== Task execution =====
try:
    print("Starting in 3 s … moving to pick pose.")
    visualize_end_effector()

    print("Moving to pick pose …")
    t0 = time.time()
    path_to_pick = plan_path(start_angles, pick_angles, "pick_history.xlsx")
    print(f"Elapsed planning time: {time.time() - t0:.4f} s")
    execute_path(path_to_pick)

    print("Closing gripper …")
    control_gripper(close=True)

    print("Attaching object …")
    cid = attach_object_to_gripper(object_id)

    print("Moving to place pose …")
    t0 = time.time()
    path_to_place = plan_path(pick_angles, place_angles, "place_history.xlsx")
    print(f"Elapsed planning time: {time.time() - t0:.4f} s")
    execute_path(path_to_place)

    print("Opening gripper …")
    control_gripper(close=False)

    print("Detaching object …")
    detach_object_from_gripper(cid)

    reset_arm_to_start()

except Exception as e:
    print(f"Error during task execution: {e}")

finally:
    p.stopStateLogging(log_id)
    print(f"Video saved to: {os.path.abspath(video_path)}")

# ===== Idle loop =====
print("Task completed. Press Ctrl+C to exit …")
while True:
    try:
        p.stepSimulation()
        time.sleep(0.01)
    except KeyboardInterrupt:
        print("Simulation ended.")
        break

p.disconnect()