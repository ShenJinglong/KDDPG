import functools

import tensorflow as tf

hidden_layer = functools.partial(
    tf.keras.layers.Dense,
    activation='relu'
)

def create_critic_network(input_shape, reward_num):
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)
    out = hidden_layer(150)(inputs)
    out = hidden_layer(150)(out)
    outputs = tf.keras.layers.Dense(reward_num, activation=None)(out)
    return tf.keras.Model(inputs, outputs)

class Critic():
    def __init__(self, agent) -> None:
        # ob_size = 1 if len(env.time_step_spec().observation.shape)==0 else env.time_step_spec().observation.shape[0]
        # ac_size = 1 if len(env.action_spec().shape)==0 else env.action_spec().shape[0]
        # self.__input_shape = (ob_size+ac_size,)
        # self.__reward_num = 1 if len(env.time_step_spec().reward.shape)==0 else env.time_step_spec().reward.shape
        self.__input_shape = (1, 15)
        self.__reward_num = 5
        self.__tau = 1e-3

        self.__net = create_critic_network(self.__input_shape, self.__reward_num)
        self.__target_net = create_critic_network(self.__input_shape, self.__reward_num)
        self.__target_net.set_weights(self.__net.get_weights())

        self.__agent = agent

        self.__optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def get_net(self):
        return self.__net

    def get_target_net(self):
        return self.__target_net

    def forward_net(self, s, a):
        return self.__net(tf.concat([s, a], axis=2))

    def forward_target_net(self, s, a):
        return self.__target_net(tf.concat([s, a], axis=2))

    def update_weights(self, data):
        batch_data = data['data']
        s_ = tf.convert_to_tensor([item[0] for item in batch_data], dtype=tf.float32)
        a_ = tf.convert_to_tensor([item[1] for item in batch_data], dtype=tf.float32)
        r_ = tf.convert_to_tensor([item[2] for item in batch_data], dtype=tf.float32)
        s_n = tf.convert_to_tensor([item[3] for item in batch_data], dtype=tf.float32)
        batch_prob = tf.convert_to_tensor(data['prob'], dtype=tf.float32)
        buffer_size = data['metadata']['buffer_size']
        batch_size = data['metadata']['batch_size']

        gamma = 0.9

        crt = self.forward_net
        act = self.__agent.actor.forward_target_net

        with tf.GradientTape() as tape:
            loss_of_transitions = tf.reduce_sum((r_ + gamma * crt(s_n, act(s_n)) - crt(s_, a_))**2, axis=2)
            weight_of_transitions = tf.convert_to_tensor([1 / (batch_prob * buffer_size)], dtype=tf.float32)
            loss = tf.matmul(weight_of_transitions, loss_of_transitions) / batch_size
            print(loss)

        grads = tape.gradient(loss, self.__net.trainable_variables)
        self.__optimizer.apply_gradients(zip(grads, self.__net.trainable_variables))
        self.__soft_update()

    def __soft_update(self):
        target_params = self.__target_net.get_weights()
        params = self.__net.get_weights()
        for index in range(len(target_params)):
            target_params[index] = target_params[index] * (1 - self.__tau) + params[index] * self.__tau
        self.__target_net.set_weights(target_params)
        self.__net.set_weights(params)