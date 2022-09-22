## Readme file for the Source Code of ' A Framework for Zero-Shot Policy Transfer through Observation Mapping and Behavior Cloning'

Overview of the paper:

Despite recent progress in Reinforcement Learning for robotics applications, many tasks remain prohibitively difficult to solve because of the expensive interaction cost. Transfer learning helps mitigate this problem by transferring knowledge learned in a source task domain, reducing the training time in the target domain. Sim2Real transfer helps transfer knowledge from a simulated robotic domain as a source to a physical target domain. This reduces the time required to train a task in the physical world, where the cost of interactions is high. However, most existing approaches assume exact correspondence in the task structure and the physical properties in the two domains. In this work, we propose a framework for Zero-Shot Policy Transfer between two domains through Observation Mapping and Behavior Cloning. We use Generative Adversarial Networks (GANs) along with a cycle-consistency loss to map the observations between the source and target domains, and later use this learned mapping to clone the successful source task behavior policy to the target domain. We observe successful zero-shot transfer with a limited number of target task interactions, and in cases where the source and target task are semantically dissimilar.


# First, to train a source policy:

`$ python PPO_source/train.py`

# To test the learned policy on the source task: 

`$ python PPO_source/test.py`

# To train CycleGAN for image translation:

`$ cd Real_robot_CycleGAN/pytorch-CycleGAN-and-pix2pix`
`$ python train.py --dataroot ./datasets/turtlebot --name turtlebot_cyclegan --model cycle_gan`

The image datasets need to be in the folder `./datasets/turtlebot`

#  Then, generate trajectories for behavior cloning:
Copy the CycleGAN output from Real_robot_CycleGAN/pytorch-CycleGAN-and-pix2pix/checkpoints to data_gan/checkpoints
`$ cd data_gen`
`$ python main.py --collect --name turtlebot_cyclegan --dataroot ./datasets/turtlebot_test --dataset_mode unaligned --model cycle_gan`

# Finally, clone the policy given the trajectories:

Copy the trajectory pickle file from data_gen/trajectories to SB_BC/trajectories

`$ python SB_BC/BC2.py`

#  And then test the cloned policy on target task:

`$ python SB_BC/test.py`

#  To set up TurtleBot2 ROS stack, `$ cd turtlebot_transfer_learning`
