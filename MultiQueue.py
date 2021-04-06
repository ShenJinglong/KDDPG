import numpy as np
import tensorflow as tf

from collections import deque

"""
    @brief: 存在队列的packet对象
"""
class Packet():
    """
        @brief: 初始化，一个packet刚进队的时候，它的hol delay为0
    """
    def __init__(self) -> None:
        self.__hol_delay = 0

    """
        @brief: 每个时隙调用一次，其hol delay加一
    """
    def step(self):
        self.__hol_delay += 1
    
    """
        @brief: 返回这个packet的hol delay
        @return: hol delay | dtype = int
    """
    def get_HoL_delay(self):
        return self.__hol_delay

"""
    @brief: 管理各个用户的队列
"""
class MultiQueue():
    """
        @brief: 初始化
        @param [num_user]: 环境中用户的数量，对应了队列的数量
        @param [queue_size]: 队列最大长度
    """
    def __init__(self, num_user, queue_size) -> None:
        self.__num_user = num_user
        self.__queue_size= queue_size
        self.__queues = [deque(maxlen=int(self.__queue_size)) for _ in range(self.__num_user)]
        self.__D_min = 5 # 最小 HoL delay
        self.__D_max = 7 # 最大 HoL delay
        self.__arrival_prob = 0.1 # 包的到达概率

    """
        @brief: 更新各个队列
        @param [action]: 用户的调度
        @return: 这个地方返回了两个值
                    delay_discriber: 这是一个包含0，1的tensor，只有一个用户的HoL delay在[Dmin, Dmax]里面，而且这个用户被调度到的时候才是1，其它都是0
                                        | type == tf.Tensor() | shape == (1, 5) | dtype == tf.float32 
                    delay_recorder: 记录了各个用户的HoL delay
    """
    def step(self, action):
        delay_discriber = []
        delay_recorder = []
        for index, queue in enumerate(self.__queues):
            if action[0][index] == 1:
                try:
                    packet = queue.popleft()
                    hol_delay = packet.get_HoL_delay()
                    delay_recorder.append(hol_delay)
                    if hol_delay >= self.__D_min and hol_delay <= self.__D_max:
                        delay_discriber.append(1)
                    else:
                        delay_discriber.append(0)
                except IndexError:
                    delay_recorder.append(0)
                    delay_discriber.append(0)
            else:
                try:
                    delay_recorder.append(queue[0].get_HoL_delay())
                except IndexError:
                    delay_recorder.append(0)
                delay_discriber.append(0)

            self.__update_queue(queue)

            if np.random.uniform() < self.__arrival_prob:
                queue.append(Packet())
        return tf.convert_to_tensor([delay_discriber], dtype=tf.float32), tf.convert_to_tensor([delay_recorder], dtype=tf.float32)

    """
        @brief: 重置各个队列（清空）
    """
    def reset(self):
        self.__queues = [deque(maxlen=int(self.__queue_size)) for _ in range(self.__num_user)]

    """
        @brief: 更新队列里各个 packet 所记录的 HoL delay
        @param [queue]: 要更新的队列
    """
    def __update_queue(self, queue):
        for packet in queue:
            packet.step()
        
    """
        @brief: 打印各个队列
    """
    def disp(self):
        print('---------------------------------------------')
        data = []
        for queue in self.__queues:
            queue_data = []
            for packet in queue:
                queue_data.append(packet.get_HoL_delay())
            data.append(queue_data)
        for item in data:
            print(item)

    """
        @brief: 返回 HoL delay 上界
    """
    def get_maximum_delay_bound(self):
        return self.__D_max

    """
        @brief: 返回 HoL delay 下界
    """
    def get_minimum_delay_bound(self):
        return self.__D_min

if __name__ == '__main__':
    queue_m = MultiQueue(5, 10000)
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[0, 0, 0, 0, 0]], dtype=tf.int64)))
    queue_m.disp()
    print('===========================================')
    print(queue_m.step(tf.convert_to_tensor([[1, 1, 1, 1, 1]], dtype=tf.int64)))
    queue_m.disp()
    