from baselines.common import set_global_seeds
from baselines.common.misc_util import boolean_flag
from baselines.common.schedules import LinearSchedule
import tensorflow as tf
import time

from qmap.models import ConvDeconvMap, ConvMlp
from qmap.q_map_dqn_agent import Q_Map_DQN_Agent
from qmap.replay_buffers import DoublePrioritizedReplayBuffer
from qmap.custom_mario import CustomSuperMarioAllStarsEnv
from qmap.wrappers import PerfLogger
n_steps = int(1000)
seed=0
model_save=100000
path="qmap_results"
env = CustomSuperMarioAllStarsEnv(screen_ratio=4, coords_ratio=8, use_color=False, use_rc_frame=False, stack=3, frame_skip=2, action_repeat=4)
coords_shape = env.coords_shape
set_global_seeds(seed)
env.seed(seed)
task_gamma = 0.99
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.__enter__()
################### comment this for no qmap ###################
q_map_model = ConvDeconvMap(
        convs=[(32, 8, 2), (32, 6, 2), (64, 4, 2)],
        middle_hiddens=[1024],
        deconvs=[(64, 4, 2), (32, 6, 2), (env.action_space.n, 4, 1)],
        coords_shape=coords_shape,
        dueling=True,
        layer_norm=True,
        activation_fn=tf.nn.elu
    )
q_map_random_schedule = LinearSchedule(schedule_timesteps=n_steps, initial_p=0.1, final_p=0.05)
######################################################################

##############comment this for no DQN#################################
dqn_model = ConvMlp(
        convs=[(32, 8, 2), (32, 6, 2), (64, 4, 2)],
        hiddens=[1024],
        dueling=True,
    )
exploration_schedule = LinearSchedule(schedule_timesteps=n_steps, initial_p=1.0, final_p=0.05)
########################################################################
double_replay_buffer = DoublePrioritizedReplayBuffer(int(5e5), alpha=0.6, epsilon=1e-6, timesteps=n_steps, initial_p=0.4, final_p=1.0)
agent_name = '{}M'.format(n_steps//int(1e6))
agent = Q_Map_DQN_Agent(
    observation_space=env.observation_space,
    n_actions=env.action_space.n,
    coords_shape=env.unwrapped.coords_shape,
    double_replay_buffer=double_replay_buffer,
    task_gamma=task_gamma,
    exploration_schedule=exploration_schedule,
    seed=seed,
    path=path,
    learning_starts=1000,
    train_freq=4,
    print_freq=1,
    env_name=env.unwrapped.name,
    agent_name=agent_name,
    renderer_viewer=False,
    dqn_q_func=dqn_model,
    dqn_lr=1e-4,
    dqn_optim_iters=1,
    dqn_batch_size=32,
    dqn_target_net_update_freq=1000,
    dqn_grad_norm_clip=1000,
    dqn_model_save_freq=model_save,
    q_map_model=q_map_model,
    q_map_random_schedule=q_map_random_schedule,
    q_map_greedy_bias=0.5,
    q_map_timer_bonus=0.5, # 50% more time than predicted
    q_map_lr=3e-4,
    q_map_gamma=0.9,
    q_map_n_steps=1,
    q_map_batch_size=32,
    q_map_optim_iters=1,
    q_map_target_net_update_freq=1000,
    q_map_min_goal_steps=15,
    q_map_max_goal_steps=30,
    q_map_grad_norm_clip=1000
)


env = PerfLogger(env, agent.task_gamma, agent.path)

done = True
episode = 0
score = None
best_score = -1e6
best_distance = -1e6
previous_time = time.time()
last_ips_t = 0

for t in range(n_steps+1):
    if done:
        new_best = False
        if episode > 0:
            if score >= best_score:
                best_score = score
                new_best = True
            distance = env.unwrapped.full_c
            if distance >= best_distance:
                best_distance = distance
                new_best = True

        if episode > 0 and (episode < 50 or episode % 10 == 0 or new_best):
            current_time = time.time()
            ips = (t - last_ips_t) / (current_time - previous_time)
            print('step: {} IPS: {:.2f}'.format(t+1, ips))
            name = 'score_%08.3f'%score + '_distance_' + str(distance) + '_steps_' + str(t+1) + '_episode_' + str(episode)
            agent.renderer.render(name)
            previous_time = current_time
            last_ips_t = t
        else:
            agent.renderer.reset()

        episode += 1
        score = 0

        ob = env.reset()
        ac = agent.reset(ob)

    ob, rew, done, _ = env.step(ac)
    score += rew

    ac = agent.step(ob, rew, done)
