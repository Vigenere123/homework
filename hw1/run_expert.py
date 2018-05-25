#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        obs_shape = expert_data['observations'].shape
        actions_shape = expert_data['actions'].shape
        print('observations shape:{}'.format(obs_shape))
        print('actions shape:{}'.format(actions_shape))

        # define placeholders
        X = tf.placeholder(shape=(None, expert_data['observations'].shape[1]), dtype=tf.float32)
        Y = tf.placeholder(shape=(None, expert_data['actions'].shape[-1]), dtype=tf.float32)

        # define layers
        l1 = tf.layers.dense(X, 60, activation=tf.nn.relu)
        l2 = tf.layers.dense(l1, 60, activation=tf.nn.relu)
        l3 = tf.layers.dense(l2, 60, activation=tf.nn.relu)
        output = tf.layers.dense(l3, Y.shape[-1], activation=None)

        cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=Y, predictions=output))
        optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
    
        tf_util.initialize()
        for step in range(10000):
            _, val = tf_util.get_session().run([optimizer, cost], feed_dict = {X: expert_data['observations'], Y: np.reshape(expert_data['actions'], (actions_shape[0], actions_shape[-1]))})

            if step % 100 == 0:
                print("step: {}, value: {}".format(step, val))
        
        # Test out our trained network on the same environment and report results
        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            print(obs.shape)
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = tf_util.get_session().run(output, feed_dict = {X: obs[None,:]})
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
if __name__ == '__main__':
    main()
