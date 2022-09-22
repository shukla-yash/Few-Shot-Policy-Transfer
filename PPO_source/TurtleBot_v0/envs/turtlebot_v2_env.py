import math
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import copy
import pybullet as p

class TurtleBotV2Env(gym.Env):
	def __init__(self, map_width=None, map_height=None, items_id=None, items_quantity=None):
		# super(TurtleBotV2Env, self).__init__()

		# p.connect(p.GUI, options='--background_color_red=.02 --background_color_green=0.02 --background_color_blue=0.02')		

		p.connect(p.GUI)
		# p.connect(p.SHARED_MEMORY)
		# p.connect(p.DIRECT)

		self.map_width = 2.0
		self.map_height = 2.0

		self.reward_step  = -1
		self.reward_done = 1000

		self.reward_hit_wall = -10

		self.time_per_episode = 300

		# low = np.zeros(self.half_beams*2*len(self.object_types) + 3)
		# high = np.ones(self.half_beams*2*len(self.object_types) + 3)
		self.img_w, self.img_h = 100, 100
		# self.observation_shape = (self.img_h, self.img_w, 3)
		self.observation_shape = (1,3,self.img_h, self.img_w)		
		self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
											high = np.ones(self.observation_shape),
											dtype = np.float64)
		# self.observation_space = spaces.Box(low, high, dtype = float)
		self.action_space = spaces.Discrete(4) # right, left, forward, backward
		self.num_envs = 1
		self.reset_time = 0


	
	def reset(self):

		# print("reset called: ", self.reset_time)
		self.reset_time += 1
		p.resetSimulation()
		p.setGravity(0,0,-10)
		# p.setTimeStep(0.01)

		offset = [0.5,0.4,0]
		self.turtle = p.loadURDF("turtlebot.urdf",offset)
		self.plane = p.loadURDF("plane.urdf")
		self.trees = []
		while True:
				self.obj_x = np.random.uniform(low = -self.map_width+0.2, high = self.map_width-0.2)
				self.obj_y = np.random.uniform(low = -self.map_width+0.2,high = self.map_width-0.2)
				if self.obj_x > -0.1 and self.obj_x < -0.1 and self.obj_y > -0.1 and self.obj_y < -0.1:
						continue
				else:
						self.trees.append(p.loadURDF("boston_box.urdf", basePosition=[self.obj_x, self.obj_y,0], useFixedBase=True))                
						break

		p.loadURDF("wall.urdf", basePosition=[self.map_width,0,0], baseOrientation=[0,0,0,1], useFixedBase=True, flags=0)
		p.loadURDF("wall.urdf", basePosition=[0,self.map_height,0], baseOrientation=[0.707,0.707,0,0], useFixedBase=True, flags=0)
		p.loadURDF("wall.urdf", basePosition=[-self.map_width,0,0], baseOrientation=[0,0,0,1], useFixedBase=True, flags=0)
		p.loadURDF("wall.urdf", basePosition=[0,-self.map_height,0], baseOrientation=[0.707,0.707,0,0], useFixedBase=True, flags=0)

		p.setRealTimeSimulation(0)

		self.env_step_counter = 0 

		obs = self.get_observation()

		return obs

	def step(self, action):

		basePos, baseOrn = p.getBasePositionAndOrientation(self.turtle)
		eulerOrn = p.getEulerFromQuaternion(baseOrn)
		rot_angle = eulerOrn[2]
		reward = self.reward_step
		done = False

		forward = 0
		turn = 0
		speed = 10
		rightWheelVelocity = 0
		leftWheelVelocity = 0
		object_removed = 0
		index_removed = 0

		if action == 0: # Turn right
			turn = 0.5
			rightWheelVelocity = -turn*speed
			leftWheelVelocity = turn*speed
			# self.nav_count += 1
		elif action == 1: # Turn left
			turn = 0.5
			rightWheelVelocity = turn*speed
			leftWheelVelocity = -turn*speed
			# self.nav_count += 1

		elif action == 2: #Move forward
			x_new = basePos[0] + 0.05*np.cos(rot_angle)
			y_new = basePos[1] + 0.05*np.sin(rot_angle)
			forward = 1
			# if abs(self.obj_x - x_new) < 0.05:
			# 	if abs(self.obj_y - y_new) < 0.05:
			# 		forward = 0

			# if (abs(abs(x_new) - abs(self.width/2)) < 0.05) or (abs(abs(y_new) - abs(self.height/2)) < 0.05):
			# 	reward = self.reward_hit_wall
			# 	forward = 0

			rightWheelVelocity = forward*speed
			leftWheelVelocity = forward*speed

		elif action == 3: #Move backward
			x_new = basePos[0] + 0.05*np.cos(rot_angle)
			y_new = basePos[1] + 0.05*np.sin(rot_angle)
			forward = 1
			# if abs(self.obj_x - x_new) < 0.05:
			# 	if abs(self.obj_y - y_new) < 0.05:
			# 		forward = 0

			# if (abs(abs(x_new) - abs(self.width/2)) < 0.05) or (abs(abs(y_new) - abs(self.height/2)) < 0.05):
			# 	reward = self.reward_hit_wall
			# 	forward = 0

			rightWheelVelocity = -forward*speed
			leftWheelVelocity = -forward*speed


		for i in range(70):
			p.setJointMotorControl2(self.turtle,0,p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000)
			p.setJointMotorControl2(self.turtle,1,p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000)
			p.stepSimulation()

		reward, done = self.check_achieved_goal()

		self.env_step_counter += 1

		obs = self.get_observation()

		return obs, reward, done, {}

	def get_observation(self):

		agent_pos, agent_orn = p.getBasePositionAndOrientation(self.turtle)        
		yaw = p.getEulerFromQuaternion(agent_orn)[-1]
		xA, yA, zA = agent_pos
		zA = zA + 0.31 # make the camera a little higher than the robot
		distance = 100000
		# compute focusing point of the camera
		xB = xA + math.cos(yaw) * distance
		yB = yA + math.sin(yaw) * distance
		zB = zA

		view_matrix = p.computeViewMatrix(
						cameraEyePosition=[xA, yA, zA],
						cameraTargetPosition=[xB, yB, zB],
						cameraUpVector=[0, 0, 1.0]
						)

		projection_matrix = p.computeProjectionMatrixFOV(
								fov=90, aspect=1.5, nearVal=0.02, farVal=3.5)

		imgs = p.getCameraImage(self.img_w, self.img_h,
								view_matrix,
								projection_matrix, shadow=True,
								renderer=p.ER_BULLET_HARDWARE_OPENGL)

		obs = np.divide(np.asarray(imgs[2]), 255)
		obs = np.swapaxes(obs,0,2)
		obs = np.expand_dims(obs, axis=0)
		return obs[:,0:3,:,:]
		# return obs[:,:,0:3]

	def check_achieved_goal(self):
		agent_pos, agent_orn = p.getBasePositionAndOrientation(self.turtle)        
		yaw = p.getEulerFromQuaternion(agent_orn)[-1]
		xA, yA, zA = agent_pos
		eulerOrn = p.getEulerFromQuaternion(agent_orn)
		rot_angle = eulerOrn[2]
		x_temp = agent_pos[0] + np.cos(rot_angle)
		y_temp = agent_pos[1] + np.sin(rot_angle)
		vec_1 = [x_temp-xA, y_temp-yA]
		vec_2 = [self.obj_x-xA, self.obj_y-yA]
		unit_vector_1 = vec_1 / np.linalg.norm(vec_1)
		unit_vector_2 = vec_2 / np.linalg.norm(vec_2)
		dot_product = np.dot(unit_vector_1, unit_vector_2)
		angle = np.arccos(dot_product)
		angle_deg = angle*57.2958
		# time.sleep(0.2)
		if (np.sqrt((xA-self.obj_x)*(xA-self.obj_x)+(yA-self.obj_y)*(yA-self.obj_y)) < 0.37) and (angle_deg < 15 or angle_deg>345):
			reward = 1000
			done = True
			# print("Done true")
		else:
			reward = -1
			done = False

		
		return reward, done		


	def close(self):
		return

