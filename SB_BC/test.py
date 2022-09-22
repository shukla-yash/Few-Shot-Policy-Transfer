'''
Imports
'''
from ast import While
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import pickle
import TurtleBot_v0
from torch.distributions import Categorical
import time
from torch.utils.data import DataLoader, TensorDataset



# infile = open('/media/yash/SSHD/Robotics/Current_Work/Curriculum_Behavior_Cloning/PPO-GAIL-cartpole/PPO/trajectories/turtlebot_GAN.pickle','rb')
# infile = open('trajectories/turtlebot_GAN_3500.pickle','rb')

# demos = pickle.load(infile)

# # demos = np.load('expert_cartpole.npz', mmap_mode='r')
# data_in = demos['states']
# data_out = demos['actions']

device = torch.device('cpu')
if(torch.cuda.is_available()):
    # torch.cuda.set_device(0)     
    device = torch.device('cuda') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


'''
Define BC Model as NN

Specs:
NN: 3 layers (4 each cells with ReLu) and Sigmoid on Output
Loss: BCE (Binary Cross Entropy)
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(4, 4)
        # self.fc2 = nn.Linear(4, 4)
        # self.fc3 = nn.Linear(4, 1)

        self.actor = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 3, stride = 1),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(64,64, kernel_size = 3, stride = 1),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        nn.Flatten(),   
                        nn.Linear(33856,512),
                        nn.ReLU(),
                        nn.Linear(512, 64),
                        nn.ReLU(),
                        nn.Linear(64,4)
                        # nn.Softmax(dim=-1)
                        )            


    def forward(self, state):
        action_probs = self.actor(state)
        # dist = Categorical(action_probs)

        # action = dist.sample()
        # action_logprob = dist.log_prob(action)
        return action_probs
        # return action.detach()
        # return action_logprob.detach()


net = Net().to(device)
model = net
loss_arr = []
'''
Train BC Model
'''
learnr = 0.003
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learnr, weight_decay=0.01)
# learning_rate = [0.01]

# data_in_squeezed = np.squeeze(np.array(data_in),1)
# data_in_torch = torch.from_numpy(data_in_squeezed).to(torch.float32).to(device)

# data_out_torch = torch.from_numpy(np.array(data_out)).to(device)

# # data_out_oh = F.one_hot(torch.from_numpy(np.array(data_out)), num_classes=4)

# train_dataset = TensorDataset(data_in_torch, data_out_torch)
# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# train_features, train_labels = next(iter(train_dataloader))


# num_epochs = 8000

# for epoch in range(num_epochs):
#     for xs, ys in iter(train_dataloader):
#         y_pred = model(xs)
#         loss = criterion(y_pred, ys)
#         print(epoch, loss.item())
#         loss_arr.append(loss.item())
#         # model.zero_grad()
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         # with torch.no_grad:
#     if len(loss_arr)> 50 and np.mean(loss_arr[-50:]) < 0.05:
#         print("Early stopping at epoch: ", epoch)
#         break

num_epochs = 8000
checkpoint_path = "ds_BC_lr={}_epochs={}.pth".format(learnr, num_epochs)
# torch.load(model.state_dict(), checkpoint_path)
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

'''
Render BC Agent and Generate Gifs
'''
env = gym.make('TurtleBot-v2')

for eps in range(100):
    obs = env.reset()
    
    while True:
        obs = torch.from_numpy(obs).to(torch.float32).to(device)
        action = model.forward(obs).detach()
        if action[0][0].item() > 0.5:
            action_to_take = 0    
        elif action[0][1].item() > 0.5:
            action_to_take = 1
        elif action[0][2].item() > 0.5:
            action_to_take = 2
        elif action[0][3].item() > 0.5:
            action_to_take = 3
        else:
            action_to_take = env.action_space.sample()

        obs, reward, done, info = env.step(action_to_take)
    #     time.sleep(0.5)
        if done:
            break
