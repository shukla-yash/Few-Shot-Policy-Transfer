from agent import Agent
from ppo import PPO
import torch
import argparse
import pickle
import TurtleBot_v0
import gym
from datagan import create_dataset
from models import create_model
import time
from PIL import Image
import numpy as np
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)



def collect(args):

    args.num_threads = 0   # test code only supports num_threads = 0
    args.batch_size = 1    # test code only supports batch_size = 1
    args.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    args.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    args.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # dataset = create_dataset(args)  # create a dataset given args.dataset_mode and other argsions
    model = create_model(args)      # create a model given args.model and other argsions
    model.setup(args)               # regular setup: load and print networks; create schedulers

    env_name = "TurtleBot-v2"
    has_continuous_action_space = False
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = False              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    env = gym.make(env_name, start_cond = 1, end_cond = 4)
    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 15      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    # agent = Agent()
    # agent.load_weights()

    # dict of arrays
    memory = {'states': [], 'actions': [], 'rewards': [], 'terminals': []}

    rewards = []
    trajectories = 0
    img_num = 0
    while trajectories < args.n_traj:
        ep_reward = 0
        state = env.reset()
        # reward = 
        # state, reward, action, terminal = agent.new_random_game()
        ep_memory = {'state': [], 'action': [], 'reward': [], 'terminal': []}
        t = 0
        while True:
            t+=1
            obs1 = np.squeeze(state, axis=0)
            obs1 = np.swapaxes(obs1,0,2)
            obs = np.multiply(obs1, 255)

            # print('obs shape: ', obs.shape)
            obs = obs.astype('uint8')
            img = Image.fromarray(obs, 'RGB')
            image_path = "datasets/turtlebot_test/testA/input.png"
            img.save(image_path)
            image_path = "datasets/turtlebot_test/testB/input.png"
            img.save(image_path)            
            dataset = create_dataset(args)  # create a dataset given opt.dataset_mode and other options
            for i, data in enumerate(dataset):
                model.set_input(data)  # unpack data from datagan loader
                model.test()           # run inference
                stateGAN = model.get_current_visuals()['fake_B']  # get image results

            out = tensor2im(stateGAN)
            image_pil = Image.fromarray(out)
            im1 = image_pil.resize((100,100), Image.BICUBIC)
            im_arr = np.array(im1)
            # for row in range(obs.shape[0]):
            #     for col in range(obs.shape[1]):
            #             if obs[row][col][1] > 150 and obs[row][col][0] < 125 and obs[row][col][2] < 125:
            #                 im_arr[row][col][1] = 166
            #                 im_arr[row][col][0] = 86
            #                 im_arr[row][col][2] = 91
            #             if obs[row][col][0] > 120 and obs[row][col][0] < 100 and obs[row][col][2] < 100:
            #                 im_arr[row][col][1] = 0
            #                 im_arr[row][col][0] = 200
            #                 im_arr[row][col][2] = 0


            im_arr = im_arr.astype('uint8')
            im1 = Image.fromarray(im_arr, 'RGB')
            image_path = "datasets/image_dataset/real_" + str(img_num) + ".png"
            im1.save(image_path)            
            image_path = "datasets/image_dataset/sim_" + str(img_num) + ".png"
            img.save(image_path)            
            img_num += 1
            final_state = np.array(im1)
            rl_state = np.divide(np.asarray(final_state), 255)
            rl_state = np.swapaxes(rl_state,0,2)
            rl_state = np.expand_dims(rl_state, axis=0)            

            files = glob.glob('datasets/turtlebot_test/testA/*')
            for f in files:
                os.remove(f)
            files = glob.glob('datasets/turtlebot_test/testB/*')
            for f in files:
                os.remove(f)                


            action = ppo_agent.select_action(state)
            new_state, reward, terminal, _ = env.step(action)            
            ep_reward += reward

            ep_memory['state'].append(rl_state)
            ep_memory['action'].append(action)
            ep_memory['reward'].append(reward)
            ep_memory['terminal'].append(terminal)
            state = new_state

            if terminal:
                if ep_reward >= args.min_reward:
                    rewards.append(ep_reward)
                    memory['states'] += ep_memory['state']
                    memory['actions'] += ep_memory['action']
                    memory['rewards'] += ep_memory['reward']
                    memory['terminals'] += ep_memory['terminal']
                    trajectories += 1
                    print("trajectory reward: {}, collected {} trajectories".format(ep_reward, trajectories))
                break
            elif t > 200:
                break
        
        f = open(args.traj_path, 'wb')
        pickle.dump(memory, f)
        f.close()
        print("trajectories saved to", args.traj_path)        

    # agent.env.close()
    avg_rew = sum(rewards) / len(rewards)
    print('avg rew: %.2f' % avg_rew)
    print('trajectories:', trajectories)
    print('states collected:', len(memory['states']))

    f = open(args.traj_path, 'wb')
    pickle.dump(memory, f)
    f.close()
    print("trajectories saved to", args.traj_path)
