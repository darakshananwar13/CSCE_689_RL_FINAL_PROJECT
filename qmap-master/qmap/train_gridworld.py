from baselines.common import set_global_seeds
from baselines.common.misc_util import boolean_flag
from baselines.common.schedules import LinearSchedule
import baselines.common.tf_util as baseline_util
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import tensorflow as tf

from qmap.models import ConvDeconvMap, MlpMap
from qmap.q_map_dqn_agent import Q_Map
from qmap.gridworld import GridWorld
from qmap.csv_logger import CSVLogger

seed=0
n_steps=1000
path_name='results'
lr=1e-4
batch=32
gamma=0.9
model_architecture="1"
target=10000
test_levels = ['level1', 'level2', 'level3']
env = GridWorld()
coords_shape = env.unwrapped.coords_shape
set_global_seeds(seed)
env.seed(seed)
test_obs = []
test_qmaps = []
image_indexes = []
n_images = 20
path = '{}/{}'.format(path_name, env.name)
for level in test_levels:
    obs_path = '{}/gridworld_obs_{}.npy'.format(path, level)
    gt_path = '{}/gridworld_gound_truth_{}.npy'.format(path, level)
    if not os.path.isfile(obs_path) or not os.path.isfile(gt_path):
        temp_env = GridWorld(level)
        temp_env.generate_ground_truth_qframes(path)
        del temp_env
    test_obs.append(np.load(obs_path))
    test_qmaps.append(np.load(gt_path))
    image_indexes.append(np.linspace(300, len(test_obs[-1]) - 300, n_images).astype(int))
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.__enter__()

if model_architecture == '1':
    q_map_model = ConvDeconvMap(
        convs=[(32, 8, 2), (32, 6, 2), (64, 4, 1)],
        middle_hiddens=[1024],
        deconvs=[(64, 4, 1), (32, 6, 2), (env.action_space.n, 4, 2)],
        coords_shape=coords_shape
    )
elif model_architecture == '2':
    q_map_model = ConvDeconvMap(
        convs=[(32, 8, 2), (32, 6, 2), (64, 4, 1)],
        middle_hiddens=[1024],
        deconvs=[(64, 4, 1), (32, 6, 2), (env.action_space.n, 8, 2)],
        coords_shape=coords_shape
    )
q_map = Q_Map(
    model=q_map_model,
    observation_space=env.observation_space,
    coords_shape=env.unwrapped.coords_shape,
    n_actions=env.action_space.n,
    gamma=gamma,
    n_steps=1,
    lr=lr,
    replay_buffer=None,
    batch_size=batch,
    optim_iters=1,
    grad_norm_clip=1000,
    double_q=True
)
baseline_util.initialize()
if not os.path.exists(path_name):
    os.mkdir(path_name)
    os.mkdir('{}/images'.format(path_name))
color_map = plt.get_cmap('inferno')

# Train.

weights = np.ones(q_map.batch_size)
completed_batch = np.zeros((q_map.batch_size, 1))
for t in range(n_steps // q_map.batch_size + 1):
    prev_frames = []
    batch_ac = []
    batch_rcw = []
    batch_frames = []
    for _ in range(q_map.batch_size):
        prev_ob = env.random_reset()
        ac = env.action_space.sample()
        ob = env.step(ac)[0]
        prev_frame, (_, _, prev_w), _, _ = prev_ob
        frames, (row, col, w), _, _ = ob
        prev_frames.append(prev_frame)
        batch_ac.append(ac)
        batch_rcw.append((row, col-w, w-prev_w))
        batch_frames.append(frames)
    prev_frames = np.array(prev_frames)
    batch_ac = np.array(batch_ac)
    batch_rcw = np.array(batch_rcw)[:, None, :]
    batch_frames = np.array(batch_frames)
    q_map._optimize(prev_frames, batch_ac, batch_rcw, batch_frames, completed_batch, weights)
    if t % target == 0:
        q_map.update_target()

    if t % 50 == 0:
        losses = []
        all_images = []
        for i_level in range(len(test_levels)):
            pred_qmaps = q_map.compute_q_values(test_obs[i_level])
            true_qmaps = test_qmaps[i_level]
            loss = np.mean((pred_qmaps - true_qmaps)**2)
            losses.append(loss)
            ob_images = np.concatenate(test_obs[i_level][image_indexes[i_level]], axis=1)
            pred_images = np.concatenate((color_map(pred_qmaps[image_indexes[i_level]].max(3))[:, :, :, :3] * 255).astype(np.uint8), axis=1)
            true_images = np.concatenate((color_map(true_qmaps[image_indexes[i_level]].max(3))[:, :, :, :3] * 255).astype(np.uint8), axis=1)
            all_images.append(np.concatenate((ob_images, true_images, pred_images), axis=0))
        img = np.concatenate(all_images, axis=0)
        #Image.fromarray(img).save('{}/images/{}.png'.format(path_name, t))
        print(t*batch, 'Losses:', *losses)