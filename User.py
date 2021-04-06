from typing import ValuesView
import numpy as np
import scipy.integrate as integrate

"""
    @brief: 极坐标转欧氏坐标
    @param [polar_coordinate]: 要转换的极坐标 | 都是用普通列表表示的坐标
    @return: 转换结果（欧氏坐标）
"""
def polar2euclid(polar_coordinate):
    return [polar_coordinate[0] * np.math.cos(polar_coordinate[1]), polar_coordinate[0] * np.math.sin(polar_coordinate[1])]

"""
    @brief: 欧氏坐标转极坐标
    @param [polar_coordinate]: 要转换的欧氏坐标 | 都是用普通列表表示的坐标
    @return: 转换结果（极坐标）
"""
def euclid2polar(euclid_coordinate):
    return [np.math.sqrt(euclid_coordinate[0]**2 + euclid_coordinate[1]**2), np.math.atan2(euclid_coordinate[1], euclid_coordinate[0])]

"""
    @brief: Q函数(对标准正态函数从指定下界积分，积分上界为无穷)
    @param [x]: 下界
"""
def qfunc(x):
    def normal_func(t):
        return np.math.exp(-0.5 * t * t) / np.math.sqrt(2 * np.math.pi)
    if x >= 0:
        return integrate.quad(normal_func, x, np.math.inf) # 本来写这一行就可以了，但是当x取-21，-22的时候，结果莫名变成了0（本来该是1），所以写成了这样的形式
    else:
        result = integrate.quad(normal_func, -x, np.math.inf)
        return (1 - result[0], result[1])

"""
    @brief: 用户对象
"""
class User():
    """
        @brief: 初始化
    """
    def __init__(self):
        self.polar_position = [np.random.uniform(0, 100), np.random.uniform(0, np.math.pi * 2)] # 用户的当前位置（欧式坐标）
        self.euclid_position = polar2euclid(self.polar_position) # 用户的当前位置（极坐标）
        self.polar_direction = [5., np.random.uniform(0, np.math.pi * 2)] # 用户的速度与方向（极坐标），矢量长度表速度，方向表用户运动方向
        self.euclid_direction = polar2euclid(self.polar_direction) # 用户的速度与方向（欧氏坐标），矢量长度表速度，方向表用户运动方向

        self.__RB_num = 1 # 用户的最优资源块
        self.__BS_trp_spectral_density = 10**-1 # W/Hz # 基站发送功率谱密度
        self.__noise_spectral_density = 10**-12 # W/Hz # 噪声功率谱密度
        self.__RB_bandwidth = 180000 # Hz # 一个RB的频谱带宽
        self.__SNR = 0 # 用户的接收信噪比

        self.__L = 32 * 8 # bits # 一个packet的长度
        self.__TTI = 125 * 10**-6 # s # 一个time slot的长度
        self.__W = 180 * 1000 # Hz # 一个RB的频谱带宽 （和 self.__RB_bandwidth 一样，写岔了...）
        self.__error_prob_max = 10**-5 # 可以接受的最大packet发送错误概率
        
        self.__update_RB_num() # 更新 self.__SNR 和 self.__RB_num

    """
        @brief: 更新SNR, 还没有考虑小尺度的衰落, 路损可能有问题
    """
    def __update_SNR(self):
        BS_transmit_power = self.__BS_trp_spectral_density * self.__RB_num * self.__RB_bandwidth # 基站发送功率
        noise_power = self.__noise_spectral_density * self.__RB_num * self.__RB_bandwidth # 噪声功率
        distance_to_BS = self.polar_position[0]
        path_loss = 45 + 30 * np.math.log10(distance_to_BS) # 路径损耗
        received_power = BS_transmit_power / (10**(path_loss/10)) # 10**((10 * np.math.log10(BS_transmit_power) - path_loss)/10) # 接收功率
        self.__SNR = received_power / noise_power

    """
        @brief: 更新最优RB_num，参考论文公式11，12
    """
    def __update_RB_num(self):
        error_prob = 1
        self.__RB_num = 0
        while error_prob > self.__error_prob_max:
            self.__RB_num += 1
            self.__update_SNR()
            q_lower_bound = ( -self.__L * np.math.log(2) + self.__TTI * self.__W * self.__RB_num * np.math.log(1 + self.__SNR) ) \
                                / np.math.sqrt( self.__TTI * self.__W * self.__RB_num * (1 - 1 / (1 + self.__SNR)**2) )
            error_prob = qfunc(q_lower_bound)[0]
        # print(q_lower_bound, error_prob)

    """
        @brief: 用户向前移动 time_elapsed 的时间，触壁会反弹
        @param [time_elapsed]: 用户所经过的时间
    """
    def __move(self, time_elapsed):
        distance = self.polar_direction[0] * time_elapsed
        pose_d = polar2euclid([distance, self.polar_direction[1]])
        self.euclid_position[0] += pose_d[0]
        self.euclid_position[1] += pose_d[1]

        self.polar_position = euclid2polar(self.euclid_position)

        if self.polar_position[0] > 100:
            normal_dir = polar2euclid([1, self.polar_position[1]])
            dot_product = self.euclid_direction[0] * normal_dir[0] + self.euclid_direction[1] * normal_dir[1]
            polar_rho_vec = [dot_product, self.polar_position[1]]
            euclid_rho_vec = polar2euclid(polar_rho_vec)
            euclid_side_vec = [self.euclid_direction[0] - euclid_rho_vec[0], self.euclid_direction[1] - euclid_rho_vec[1]]
            self.euclid_direction[0], self.euclid_direction[1] = euclid_side_vec[0] - euclid_rho_vec[0], euclid_side_vec[1] - euclid_rho_vec[1]
            self.polar_direction = euclid2polar(self.euclid_direction)

    """
        @brief: 更新用户，包括移动和资源，信噪比等的更新
        @param [time_elapsed]: 用户所经过的时间
    """
    def step(self, time_elapsed):
        self.__move(time_elapsed)
        self.__update_RB_num()

    """
        @brief: 返回用户的信噪比
    """
    def get_SNR(self):
        return self.__SNR

    """
        @brief: 返回用户的最优RB
    """
    def get_RB_num(self):
        return self.__RB_num

    """
        @brief: 返回用户给出的奖励（不考虑调度和delay），参考论文公式11, 16, 17
        @return: 返回的奖励 | dtype=float
    """
    def get_reward(self):
        q_lower_bound = ( -self.__L * np.math.log(2) + self.__TTI * self.__W * self.__RB_num * np.math.log(1 + self.__SNR) ) \
                                / np.math.sqrt( self.__TTI * self.__W * self.__RB_num * (1 - 1 / (1 + self.__SNR)**2) )
        error_prob = qfunc(q_lower_bound)[0]
        # print(error_prob)
        try:
            return -np.math.log10(error_prob) # error prob有可能小到十分接近 0，计算机会直接表示为 0
        except ValueError:
            return 324. # 最小表示到 1e-323, 所以这里返回一个324

import matplotlib.pyplot as plt

if __name__ == '__main__':
    users = []
    for _ in range(5):
        users.append(User())

    plt.figure(figsize=(8, 6), dpi=80)
    plt.ion()
    while True:
        plt.cla()
        # print('##########################################')
        for user in users:
            user.step(1)
        for index, user in enumerate(users):
            plt.scatter(user.euclid_position[0], user.euclid_position[1])
            plt.plot([0, user.euclid_position[0]], [0, user.euclid_position[1]])
            plt.text(user.euclid_position[0], user.euclid_position[1], f'ID: {index}, SNR: {np.math.log10(user.get_SNR()):.1f}, RB: {user.get_RB_num()}, R: {user.get_reward():.1f}')
        
        thetas = np.linspace(0, np.math.pi*2, 200)
        x = 100 * np.cos(thetas)
        y = 100 * np.sin(thetas)
        plt.plot(x, y)
        plt.axis('equal')
        plt.pause(0.01)
        # input()