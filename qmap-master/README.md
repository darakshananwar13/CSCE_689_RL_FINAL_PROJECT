# Reinfrocement-Learning-Project
## All Goal Update RL using CNN

- This repositary contains the code for our project in CSCE 689 Reinforcement Learning taught in Spring 
2020 by [Prof. Guni Sharon](http://faculty.cse.tamu.edu/guni/).
- Algorithm and results presented here are based on the paper 
[Scaling All-Goals Updates in Reinforcement Learning Using Convolutional Neural Networks(https://arxiv.org/abs/1810.02927).

## Steps to follow for running the given source code 

1. Install  TensorFlow, Baselines, Gym and Gym Retro.
Please find the following snippet used for installing these in google colab.



2. Clone our repository and install the setup-tools dependencies using pip install -e command.
Please find the following snippet for the same.



3. Run train_gridworld.py in order to train for grid world. 
4. Run train_montezuma.py in order to train model for montezume game.
5. In order to run Super Mario game, we need to copy the game environment in the directory where GYM retro is installed. 
Please follow the snippet below to copy in google-colab environment.

Now, run train_mario.py for training the model for Mario game.
6. We can find the videos of the game played during training saved under “result” folder created.


