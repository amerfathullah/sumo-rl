import argparse
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
import ray
from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.tune.registry import register_env
from gym import spaces
import numpy as np
from sumo_rl import SumoEnvironment
import traci


if __name__ == '__main__':
    ray.init()

    register_env("2way-single-intersection", lambda _: SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                                                    route_file='nets/2way-single-intersection/single-intersection-gen.rou.xml',
                                                    out_csv_name='outputs/2way-single-intersection/a3c',
                                                    use_gui=False,
                                                    num_seconds=100000,
                                                    max_depart_delay=0))

    trainer = A3CTrainer(env="2way-single-intersection", config={
        "lr": 0.001,
        "no_done_at_end": True
    })
    while True:
        print(trainer.train())  # distributed training step