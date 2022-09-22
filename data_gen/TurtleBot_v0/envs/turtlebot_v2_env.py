# Wall color - (light brown); wall size - 2.5 meters x 2.5 meters; wall height - default is fine
# Ball color - Green (sphere-small)
# Goal color - red
# Background color - White
# Entrance color - light brown; entrance gap - turtlebot comfortably goes in
# Goal 0: Reach the ball; Goal 1: push ball; 
import math
from multiprocessing.process import BaseProcess
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import copy
import pybullet as p
from PIL import Image

class TurtleBotV2Env(gym.Env):
	def __init__(self, start_cond = 1, end_cond = 4):
		# super(TurtleBotV2Env, self).__init__()

		# p.connect(p.GUI)
		p.connect(p.GUI, options='--background_color_red=1 --background_color_green=1 --background_color_blue=1') # How to change background color
		# p.connect(p.SHARED_MEMORY)
		# p.connect(p.DIRECT)
		# p.connect(p.DIRECT, options='--background_color_red=.02 --background_color_green=0.02 --background_color_blue=0.02')		

		self.start_cond = start_cond
		self.end_cond = end_cond

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

		if(self.start_cond == 1): #spawn randomly in big area
				self.obj_x = 0.1
				self.obj_y = 0.3
				offset = [np.random.uniform(low = -1.6, high = -0.3),np.random.uniform(low = -1.6, high = 1.6),0]

		if(self.start_cond == 2): #spawn next to ball
				self.obj_x = 0.1
				self.obj_y = 0.3
				offset = [self.obj_x - np.random.uniform(low = 0, high = 0.5), self.obj_y- 0.4,0]
		if(self.start_cond == 3):
				self.obj_x = np.random.uniform(low = -1.5, high = 0)
				self.obj_y = np.random.uniform(low = -1.5,high = self.map_width-0.2)
				offset = [np.random.uniform(low = -0.5, high = 0),np.random.uniform(low = -0.2, high = 0.8),0]
		if(self.start_cond == 4): #spawn randomly in small area
				self.obj_x = np.random.uniform(low = -1.5, high = 0)
				self.obj_y = np.random.uniform(low = -1.5,high = self.map_width-0.2)
				offset = [np.random.uniform(low = 0.8, high = 1.5),np.random.uniform(low = -1.6, high = 1.6),0]
				

		self.turtle = p.loadURDF("turtlebot.urdf",offset)
		self.plane = p.loadURDF("plane.urdf")
		self.trees = []
		while True:
				if self.obj_x > -0.1 and self.obj_x < -0.1 and self.obj_y > -0.1 and self.obj_y < -0.1:
					continue
				else:
					# self.trees.append(p.loadURDF("boston_box.urdf", basePosition=[self.obj_x, self.obj_y,0], useFixedBase=True))
					self.trees.append(p.loadURDF("sphere_small.urdf", basePosition=[self.obj_x, self.obj_y,0.2], useFixedBase=False))
					break
		# print("low: ", -self.map_width+0.45)
		# print("high: ", self.map_width-0.45)
		
		# print("object x: ", self.obj_x)
		# print("object y: ", self.obj_y)

		p.loadURDF("wall.urdf", basePosition=[self.map_width,0,0], baseOrientation=[0,0,0,1], useFixedBase=True, flags=0)
		p.loadURDF("wall.urdf", basePosition=[0,self.map_height,0], baseOrientation=[0.707,0.707,0,0], useFixedBase=True, flags=0)
		p.loadURDF("wall.urdf", basePosition=[-self.map_width,0,0], baseOrientation=[0,0,0,1], useFixedBase=True, flags=0)
		p.loadURDF("wall.urdf", basePosition=[0,-self.map_height,0], baseOrientation=[0.707,0.707,0,0], useFixedBase=True, flags=0)
		p.loadURDF("innerwall_large.urdf", basePosition=[self.map_width/4,-self.map_height + 1,0], baseOrientation=[0,0,0,1], useFixedBase=True, flags=0)
		p.loadURDF("innerwall_small.urdf", basePosition=[self.map_width/4,self.map_height - 0.6,0], baseOrientation=[0,0,0,1], useFixedBase=True, flags=0)


		self.goal = p.loadURDF("goal.urdf", basePosition=[self.map_width - 0.05, np.random.uniform(low = self.map_height -3.6, high = self.map_height-0.3),0], baseOrientation=[0,0,0,1], useFixedBase=True, flags=0)



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


		for i in range(40):
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
								fov=80, aspect=1.5, nearVal=0.02, farVal=5.5)

		imgs = p.getCameraImage(self.img_w, self.img_h,
								view_matrix,
								projection_matrix, shadow=True,
								renderer=p.ER_BULLET_HARDWARE_OPENGL)

		obs = np.divide(np.asarray(imgs[2]), 255)
		obs = np.swapaxes(obs,0,2)
		obs = np.expand_dims(obs, axis=0)
		return obs[:,0:3,:,:]

	def check_contact(self):
		in_contact = False
		

		return in_contact

	def check_achieved_goal(self): # This would also depend on env_type
		if(self.end_cond == 1):#reach the ball
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
			if (np.sqrt((xA-self.obj_x)*(xA-self.obj_x)+(yA-self.obj_y)*(yA-self.obj_y)) < 0.45) and (angle_deg < 90 or angle_deg>270):
				reward = 1000
				done = True
				# print("Done true")
			else:
				reward = -1
				done = False
				# return 
		
		if(self.end_cond == 2): #contact and not obstructing
			obj_pos, obj_orn = p.getBasePositionAndOrientation(self.trees[0])
			obj_x = obj_pos[0]
			obj_y = obj_pos[1]
			if(obj_x != 0.5 and obj_y > 0.8 or obj_y < -0.1):
				reward = 1000
				done = True
				#print("Done true")		
			else:
				reward = -1
				done = False
				# return		
		if(self.end_cond == 3):# move past the entrance
			turtle_pos, goalOrn = p.getBasePositionAndOrientation(self.turtle)
			turtle_x = turtle_pos[0]

			obj_pos, obj_orn = p.getBasePositionAndOrientation(self.trees[0])
			obj_x = obj_pos[0]
			obj_y = obj_pos[1]

			if(turtle_x > 0.8 and (obj_x < 0 or obj_x > 0.8) and (obj_y < 0 or obj_y > 0.8)): #past the wall
				reward = 1000
				done = True
				# print("Done true")
			else:
				reward = -1
				done = False
				# return 
			# return
		if(self.end_cond == 4): #reach the goal 
			goalPos, goalOrn = p.getBasePositionAndOrientation(self.goal)
			obj_x = goalPos[0]
			obj_y = goalPos[1]
			agent_pos, agent_orn = p.getBasePositionAndOrientation(self.turtle)        
			yaw = p.getEulerFromQuaternion(agent_orn)[-1]
			xA, yA, zA = agent_pos
			eulerOrn = p.getEulerFromQuaternion(agent_orn)
			rot_angle = eulerOrn[2]
			x_temp = agent_pos[0] + np.cos(rot_angle)
			y_temp = agent_pos[1] + np.sin(rot_angle)
			vec_1 = [x_temp-xA, y_temp-yA]
			vec_2 = [obj_x-xA, obj_y-yA]
			unit_vector_1 = vec_1 / np.linalg.norm(vec_1)
			unit_vector_2 = vec_2 / np.linalg.norm(vec_2)
			dot_product = np.dot(unit_vector_1, unit_vector_2)
			angle = np.arccos(dot_product)
			angle_deg = angle*57.2958
			# time.sleep(0.2)
			if (np.sqrt((xA-obj_x)*(xA-obj_x)+(yA-obj_y)*(yA-obj_y)) < 0.37) and (angle_deg < 30 or angle_deg>330):
				reward = 1000
				done = True
				# print("Done true")
			else:
				reward = -1
				done = False
				# return 
    		
		return reward, done		


	def close(self):
		return

