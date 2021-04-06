
import tensorflow as tf

from User import User
from MultiQueue import MultiQueue

"""
    @brief: 仿真环境
"""
class SimuEnv():
    """
        @brief: 环境初始化
    """
    def __init__(self) -> None:
        self.__state = tf.zeros((1, 10), dtype=tf.float32) # 环境的初始状态全是0
        self.__episode_ended = False # episode 是否结束的标志位
        self.__step_counter = 0 # 记录环境跑了多少个时隙

        self.__users = [] # 5个用户的实例
        for _ in range(5):
            self.__users.append(User())
        self.__multi_queue = MultiQueue(5, 200) # 各个用户的通信队列

    """
        @brief: 重置环境，但是各个user的情况照旧，没动它，可能有问题
        @return: 返回一个time_step字典，包含了环境状态、奖励、和当前time_step是否是一个episode的最后一步
    """
    def reset(self):
        self.__state = tf.zeros((1, 10), dtype=tf.float32)
        self.__episode_ended = False
        self.__step_counter = 0
        self.__multi_queue.reset()
        return {
            'observation': self.__state, # type == tf.Tensor() | shape == (1, 10) | dtype == tf.float32
            'reward': tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.float32), # type == tf.Tensor() | shape == (1, 5) | dtype == tf.float32
            'is_last': False
        }

    """
        @brief: 返回环境的当前状态
        @return: 表示环境状态的tensor | type == tf.Tensor() | shape == (1, 10) | dtype == tf.float32
    """
    def current_state(self):
        return self.__state

    """
        @brief: 环境步进
        @param [action]: 对各个用户的调度 | type == tf.Tensor() | shape == (1, 5) | dtype == tf.float32 | 各个元素应该限制为0或1，分别表示未调度和已调度
        @return: 返回一个time_step字典，包含了环境状态、奖励、和当前time_step是否是一个episode的最后一步
    """
    def step(self, action):
        
        if self.__episode_ended: # 如果上一步是一轮episode的结尾，那么这一步就重置环境
            return self.reset()

        """
            这个地方返回了两个值
            delay_discriber: 这是一个包含0，1的tensor，只有一个用户的HoL delay在[Dmin, Dmax]里面，而且这个用户被调度到的时候才是1，其它都是0
                                | type == tf.Tensor() | shape == (1, 5) | dtype == tf.float32 
            delay_recorder: 记录了各个用户的HoL delay
        """
        delay_discriber, delay_recorder = self.__multi_queue.step(action) 

        reward = []
        RB_num = []
        for user in self.__users:
            user.step(0.000125) # 一个时隙是 125us，也就是 0.000125s
            reward.append(user.get_reward()) # 记录各个用户返回的 reward
            RB_num.append(user.get_RB_num()) # 记录各个用户的最佳 RB num
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)
        final_reward = reward * delay_discriber # 之前返回的reward没有考虑调度情况和HoL delay，这个地方乘上之前的 delay_discriber 就可以了。对于delay不在范围内，和未被调度的用户，其奖励置0

        part_1 = delay_recorder / self.__multi_queue.get_maximum_delay_bound() # 按论文公式15填充环境状态
        part_2 = tf.convert_to_tensor([RB_num], dtype=tf.float32) / 40 # 公式15里面除的是N（好像是RB num的总数？），这个地方直接除的40，不知道有问题没？
        self.__state = tf.concat([part_1, part_2], axis=1)

        self.__step_counter += 1
        if self.__step_counter == 200: # 一个 episode 跑 200 个 time slot
            self.__episode_ended = True
        return {'observation': self.__state, 'reward': final_reward, 'is_last': self.__episode_ended}
        

if __name__ == '__main__':
    env = SimuEnv()
    # print(env.reset())
    # print(env.step(tf.convert_to_tensor([[1, 0, 0, 1, 0]], tf.float32)))
    # print(env.step(tf.convert_to_tensor([[1, 0, 0, 1, 0]], tf.float32)))
    # print(env.step(tf.convert_to_tensor([[1, 0, 0, 1, 0]], tf.float32)))
    # print(env.step(tf.convert_to_tensor([[1, 0, 0, 1, 0]], tf.float32)))
    # print(env.step(tf.convert_to_tensor([[1, 0, 0, 1, 0]], tf.float32)))
    # print(env.step(tf.convert_to_tensor([[1, 0, 0, 1, 0]], tf.float32)))
    # print(env.step(tf.convert_to_tensor([[1, 0, 0, 1, 0]], tf.float32)))
    # print(env.step(tf.convert_to_tensor([[1, 0, 0, 1, 0]], tf.float32)))

    cur_step = env.reset()
    counter = 0
    while not cur_step['is_last']:
        cur_step = env.step(tf.convert_to_tensor([[1, 0, 0, 1, 0]], tf.float32))
        counter += 1
    print(counter)



