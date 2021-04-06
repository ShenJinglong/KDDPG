from DdpgAgent import DdpgAgent
from SimuEnv import SimuEnv

EPISODE_NUM = 1

if __name__ == '__main__':
    simu_env = SimuEnv()
    ddpg_agent = DdpgAgent(10000, 20)
    ddpg_agent.init_replay_buffer(simu_env)

    # ddpg_agent.update()

    for episode in range(EPISODE_NUM):
        print(f'========================EPISODE {episode}============================')
        cur_step = simu_env.reset()
        while not cur_step['is_last']:
            cur_state = simu_env.current_state()
            action = ddpg_agent.act(cur_state)
            cur_step = simu_env.step(action)
            ddpg_agent.remember(cur_state, action, cur_step['reward'], cur_step['observation'])
            ddpg_agent.update()


