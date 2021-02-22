import gym
from stable_baselines3.dqn.dqn import DQN
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import SumoEnvironment
import traci


if __name__ == '__main__':

    env = SumoEnvironment(net_file='nets/gambir/gambir.net.xml',
                                    route_file='nets/gambir/gambir.rou.xml',
                                    out_csv_name='outputs/gambir/dqn',
                                    single_agent=True,
                                    use_gui=False,
                                    num_seconds=100000,
                                    max_depart_delay=0)

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.01,
        learning_starts=0,
        train_freq=1,
        target_update_interval=100,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1
    )
    model.learn(total_timesteps=100000)

sumoBinary = "/usr/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "/home/amerfathullah/Desktop/gambir/gambir.sumocfg"]

traci.start(env)
step = 0
while step < 1000:
   traci.simulationStep()
   step += 1

traci.close()