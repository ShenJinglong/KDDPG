import random

import numpy as np
import tensorflow as tf

from collections import deque

# """
#     brief: 用target net更新的权重，后边检查一下
# """
# class WeightedReplayBuffer():
#     def __init__(self, buffer_size, batch_size, actor, critic) -> None:
#         self.__buffer = deque(maxlen=int(buffer_size))
#         self.__weights = deque(maxlen=int(buffer_size))
#         self.__batch_size = batch_size

#         self.__actor_net = actor.get_target_net()
#         self.__critic_net = critic.get_target_net()

#     def append(self, s, a, r, sn):
#         self.__buffer.append([s, a, r, sn])
#         if len(self.__weights) == 0:
#             self.__weights.append(1e-4)
#         else:
#             self.__weights.append(max(self.__weights))

#     def get_buffer(self):
#         return self.__buffer

#     def get_weights(self):
#         return self.__weights

#     def get_batch(self):
#         probs = np.array(self.__weights, dtype=np.float32) / sum(self.__weights)
#         probs[-1] = 1 - sum(probs[0:-1])
#         selected_index = np.random.choice(len(self.__buffer), size=self.__batch_size, replace=False, p=list(probs))
#         batch = {
#             "metadata": {
#                 "buffer_size": len(self.__buffer),
#                 "batch_size": self.__batch_size
#             },
#             "data": [self.__buffer[index] for index in selected_index],
#             "prob": [probs[index] for index in selected_index]
#         }
#         self.__update_weights(selected_index)
#         return batch

#     def __update_weights(self, selected_index):
#         gamma = 0.9
#         for index in selected_index:
#             transition = self.__buffer[index]
#             y_shaped = transition[2] + gamma * self.__critic_net(
#                 np.array([
#                     np.concatenate(
#                         [transition[3], self.__actor_net(np.array([transition[3]], dtype=np.float32)).numpy()[0]]
#                     )
#                 ], dtype=np.float32)
#             )
#             self.__weights[index] = sum(
#                 (  (y_shaped - self.__critic_net(np.array([np.concatenate([transition[0], transition[1]])], dtype=np.float32)))**2  )
#             ).numpy()[0]

# if __name__ == '__main__':
#     actor = Actor()
#     critic = Critic()
#     weighted_replay_buffer = WeightedReplayBuffer(10000, 2, actor, critic)
#     weighted_replay_buffer.append([1], [0], [4], [1])
#     weighted_replay_buffer.append([2], [1], [5], [2])
#     weighted_replay_buffer.append([3], [2], [6], [3])
#     weighted_replay_buffer.append([4], [3], [7], [4])
#     weighted_replay_buffer.append([5], [4], [8], [5])
#     weighted_replay_buffer.append([6], [5], [9], [6])
#     weighted_replay_buffer.append([7], [6], [0], [7])


#     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Weights Update Will Start ...')
#     print(weighted_replay_buffer.get_buffer())
#     print(weighted_replay_buffer.get_weights())
#     print(weighted_replay_buffer.get_batch())

#     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Weights Update Will Start ...')
#     print(weighted_replay_buffer.get_buffer())
#     print(weighted_replay_buffer.get_weights())
#     print(weighted_replay_buffer.get_batch())

#     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Weights Update Will Start ...')
#     print(weighted_replay_buffer.get_buffer())
#     print(weighted_replay_buffer.get_weights())
#     print(weighted_replay_buffer.get_batch())
























###############################################################################

























"""
    brief: 用target net更新的权重，后边检查一下
"""
class WeightedReplayBuffer():
    def __init__(self, buffer_size, batch_size, agent) -> None:
        self.__buffer = deque(maxlen=int(buffer_size))
        self.__weights = deque(maxlen=int(buffer_size))
        self.__batch_size = batch_size

        self.__agent = agent

    def append(self, s, a, r, sn):
        self.__buffer.append([s, a, r, sn])
        if len(self.__weights) == 0:
            self.__weights.append(1e-4)
        else:
            self.__weights.append(max(self.__weights))
        # print("Batch added ... -> ", [s, a, r, sn], self.__weights[-1], end='\n\n')

    def get_buffer(self):
        return self.__buffer

    def get_weights(self):
        return self.__weights

    def get_batch(self):
        probs = np.array(self.__weights, dtype=np.float32) / sum(self.__weights)
        front_part = list(probs[0:-1])
        tail_part = 1. - sum(front_part)
        selected_index = np.random.choice(len(self.__buffer), size=self.__batch_size, replace=False, p=front_part.append(tail_part))
        batch = {
            "metadata": {
                "buffer_size": len(self.__buffer),
                "batch_size": self.__batch_size
            },
            "data": [self.__buffer[index] for index in selected_index],
            "prob": [probs[index] for index in selected_index]
        }
        self.__update_weights(selected_index)
        return batch

    def __update_weights(self, selected_index):
        gamma = 0.9
        data = [self.__buffer[index] for index in selected_index]
        s_ = tf.convert_to_tensor([item[0] for item in data], dtype=tf.float32)
        # print('s_ -> ', s_)
        a_ = tf.convert_to_tensor([item[1] for item in data], dtype=tf.float32)
        # print('a_ -> ', a_)
        r_ = tf.convert_to_tensor([item[2] for item in data], dtype=tf.float32)
        # print('r_ -> ', r_)
        s_n = tf.convert_to_tensor([item[3] for item in data], dtype=tf.float32)
        # print('s_n -> ', s_n)

        crt = self.__agent.critic.forward_target_net
        act = self.__agent.actor.forward_target_net
        weight_of_transitions = tf.reduce_sum((r_ + gamma * crt(s_n, act(s_n)) - crt(s_, a_))**2, axis=2)
        # print('weight_of_transitions -> ', weight_of_transitions)
        for i, index in enumerate(selected_index):
            self.__weights[index] = weight_of_transitions.numpy()[i][0]

        # print(self.__weights)










