"""
 * @file            dexterous_hands/mujoco_hand.py
 * @description     
 * @author          nicewang <wangxiaonannice@gmail.com>
 * @createTime      2026-03-11
 * @lastModified    2026-03-11
 * Copyright Xiaonan (Nice) Wang. All rights reserved
"""

import mujoco
import mujoco.viewer
import time
import math

# ==============================================================================
# 1. Physical Modeling: MJCF (MuJoCo XML) Definition
# ==============================================================================
# Here we define an underactuated finger matching the paper's paradigm:
# - Spring Agonists: Joint stiffness tries to keep the finger straight (springref="0").
# - Tendon Antagonists: A spatial tendon routes through the bottom of the finger.
# - Underactuation: Only ONE motor pulls the tendon, which flexes TWO joints.
xml_string = """
<mujoco model="underactuated_finger">
    <compiler angle="degree" coordinate="local"/>
    <option gravity="0 0 -9.81" timestep="0.002"/>
    
    <!-- Define default properties to keep the XML clean -->
    <default>
        <geom type="capsule" size="0.015" rgba="0.8 0.9 0.8 1"/>
        <!-- 
             SPRING AGONIST: 
             stiffness="2.0" acts as a torsional spring trying to restore the joint.
             damping="0.1" prevents infinite oscillation.
        -->
        <joint type="hinge" axis="0 1 0" stiffness="2.0" damping="0.1" springref="0"/>
        <!-- Sites are used as routing pulleys for the tendon -->
        <site type="sphere" size="0.005" rgba="1 0 0 1"/>
    </default>

    <worldbody>
        <!-- Light source and ground plane -->
        <light pos="0 1 1" dir="0 -1 -1" diffuse="1 1 1"/>
        <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>

        <!-- Palm / Base -->
        <body name="palm" pos="0 0 0.1">
            <geom type="box" size="0.05 0.05 0.02" rgba="0.2 0.2 0.2 1"/>
            <site name="anchor_site" pos="-0.04 0 -0.025"/>
            
            <!-- Proximal Phalanx (Base joint) -->
            <body name="proximal_link" pos="0.05 0 0">
                <joint name="proximal_joint" range="-90 0" limited="true"/>
                <geom pos="0.05 0 0" size="0.015 0.05" euler="0 90 0"/>
                <!-- Tendon routing site on the palmar side (Z < 0) -->
                <site name="proximal_site" pos="0.05 0 -0.02"/>

                <!-- Distal Phalanx (Tip joint) -->
                <body name="distal_link" pos="0.1 0 0">
                    <joint name="distal_joint" range="-90 0" limited="true"/>
                    <geom pos="0.04 0 0" size="0.012 0.04" euler="0 90 0"/>
                    <!-- Tendon routing site at the fingertip -->
                    <site name="distal_site" pos="0.04 0 -0.015"/>
                </body>
            </body>
        </body>
    </worldbody>

    <!-- 
         TENDON ANTAGONIST: 
         A spatial tendon stringing through the defined sites. 
    -->
    <tendon>
        <spatial name="flexor_tendon" width="0.002" rgba="0 0 1 1">
            <site site="anchor_site"/>
            <site site="proximal_site"/>
            <site site="distal_site"/>
        </spatial>
    </tendon>

    <!-- 
         UNDERACTUATION: 
         A single motor pulling the entire tendon. 
    -->
    <actuator>
        <motor name="tendon_motor" tendon="flexor_tendon" ctrlrange="0 10" ctrllimited="true"/>
    </actuator>
</mujoco>
"""

# ==============================================================================
# 2. Simulation and Control Loop
# ==============================================================================

# Load the model and create data structure from the XML string
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

print("Starting MuJoCo simulation...")
print("Observe the red dots (tendon routing sites) and the blue line (tendon).")

# Launch the interactive passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    start_time = time.time()
    
    while viewer.is_running():
        step_start = time.time()
        
        # --- Control Logic ---
        # Generate a sinusoidal force to simulate rhythmic grasping and releasing
        # Force is strictly positive (tendons can only pull, not push)
        elapsed_time = time.time() - start_time
        target_force = 5.0 * (math.sin(elapsed_time * 2.0) + 1.0) 
        
        # Apply force to the single motor
        data.ctrl[0] = target_force
        
        # Step the physics engine forward
        mujoco.mj_step(model, data)
        
        # Sync visualization with real time
        viewer.sync()
        
        # Maintain roughly 60 FPS for the viewer
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)