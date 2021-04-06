import tensorflow as tf
import numpy as np

from WeightedReplayBuffer import WeightedReplayBuffer
from Critic import Critic
from Actor import Actor

class DdpgAgent():
    def __init__(self, replay_buffer_size, batch_size) -> None:
        self.__critic = Critic(self)
        self.__actor = Actor(self)
        self.__replay_buffer_size = int(replay_buffer_size)
        self.__batch_size = int(batch_size)
        self.__weighted_replay_buffer = WeightedReplayBuffer(self.__replay_buffer_size, self.__batch_size, self)

    def init_replay_buffer(self, env):
        env.reset()
        for _ in range(self.__batch_size):
            cur_state = env.current_state()
            action = tf.convert_to_tensor(np.random.randint(0, 2, size=(1, 5), dtype=np.int64), dtype=tf.float32)
            cur_step = env.step(action)
            self.remember(cur_state, action, cur_step['reward'], cur_step['observation'])

    @property
    def critic(self):
        return self.__critic

    @property
    def actor(self):
        return self.__actor

    def act(self, s):
        return self.__actor.forward_net(s) ###########################################

    def remember(self, s, a, r, sn):
        self.__weighted_replay_buffer.append(s, a, r, sn)

    def update(self):
        batch_data = self.__weighted_replay_buffer.get_batch()
        # print(batch_data)
        self.__critic.update_weights(batch_data)
        self.__actor.update_weights(batch_data)

if __name__ == '__main__':
    ddpg_agent = DdpgAgent(10000, 2)

    s1 = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=tf.float32)
    s2 = tf.convert_to_tensor([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype=tf.float32)
    s3 = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=tf.float32)
    s4 = tf.convert_to_tensor([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype=tf.float32)

    a1 = tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.int64)
    a2 = tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)
    a3 = tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)
    a4 = tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.int64)

    r1 = tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.float32)
    r2 = tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.float32)
    r3 = tf.convert_to_tensor([[-1, -1, -1, -1, -1]], dtype=tf.float32)
    r4 = tf.convert_to_tensor([[-1, -1, -1, -1, -1]], dtype=tf.float32)

    sn1 = tf.convert_to_tensor([[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]], dtype=tf.float32)
    sn2 = tf.convert_to_tensor([[-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5]], dtype=tf.float32)
    sn3 = tf.convert_to_tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], dtype=tf.float32)
    sn4 = tf.convert_to_tensor([[-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]], dtype=tf.float32)
    ddpg_agent.remember(s1, a1, r1, sn1)
    ddpg_agent.remember(s2, a2, r2, sn2)
    ddpg_agent.remember(s3, a3, r3, sn3)
    ddpg_agent.remember(s4, a4, r4, sn4)

    for _ in range(1000):
        ddpg_agent.update()

    print(ddpg_agent.act(s1))
    print(ddpg_agent.act(s2))
    print(ddpg_agent.act(s3))  
    print(ddpg_agent.act(s4))
    

