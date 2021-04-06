import functools

import tensorflow as tf

hidden_layer = functools.partial(
    tf.keras.layers.Dense,
    activation='relu'
)

def create_actor_network(input_shape, action_num):
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)
    out = hidden_layer(100)(inputs)
    out = hidden_layer(100)(out)
    outputs = tf.keras.layers.Dense(action_num, activation=lambda x: 0.5 * tf.keras.activations.tanh(x) + 0.5)(out)
    return tf.keras.Model(inputs, outputs)

class Actor():
    def __init__(self, agent) -> None:
        # self.__input_shape = env.time_step_spec().observation.shape
        # self.__action_num = env.action_spec().maximum - env.action_spec().minimum + 1
        # self.__critic_net = critic.get_net()
        self.__input_shape = (1, 10)
        self.__action_num = 5
        self.__tau = 1e-3

        self.__net = create_actor_network(self.__input_shape, self.__action_num)
        self.__target_net = create_actor_network(self.__input_shape, self.__action_num)
        self.__target_net.set_weights(self.__net.get_weights())

        self.__agent = agent

        self.__optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def get_net(self):
        return self.__net

    def get_target_net(self):
        return self.__target_net

    def forward_net(self, s):
        return self.__net(tf.convert_to_tensor(s, dtype=tf.float32))

    def forward_target_net(self, s):
        # print('------------------------------------------')
        # print(s)
        return self.__target_net(tf.convert_to_tensor(s, dtype=tf.float32))

    def update_weights(self, data):
        batch_data = data['data']
        s_ = tf.convert_to_tensor([item[0] for item in batch_data], dtype=tf.float32)
        batch_prob = tf.convert_to_tensor(data['prob'], dtype=tf.float32)
        buffer_size = data['metadata']['buffer_size']
        batch_size = data['metadata']['batch_size']

        crt = self.__agent.critic.forward_target_net
        act = self.forward_net

        with tf.GradientTape() as tape:         
            score_of_transitions = tf.reduce_sum(crt(s_, act(s_)), axis=2)
            weight_of_transitions = tf.convert_to_tensor([1 / (batch_prob * buffer_size)], dtype=tf.float32)
            loss = -tf.matmul(weight_of_transitions, score_of_transitions) / batch_size
            print(loss, end='\n\n')
        
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

if __name__ == '__main__':
    net1 = create_actor_network((1, 10), 5)
    net2 = create_actor_network((1, 10), 5)
    tau = 1e-3

    target_params = net1.get_weights()
    params = net2.get_weights()
    for index in range(len(target_params)):
        target_params[index] = target_params[index] * (1 - tau) + params[index] * tau
    net1.set_weights(target_params)
    net2.set_weights(params)


