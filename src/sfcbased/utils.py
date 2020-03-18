import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import csv
import collections
from ast import literal_eval
from sfcbased.model import *


class Action:
    pass


class Environment:
    pass


class NormalEnvironment(Environment):
    def get_reward(self, model: Model, sfc_index: int, decision: Decision, test_env: TestEnv):
        return 0

    def get_state(self, model: Model, sfc_index: int):
        return []


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    """
    Experience buffer class
    """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        """
        Append an experience item
        :param experience: experience item
        :return: nothing
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        Sample a batch from this buffer
        :param batch_size: sample size
        :return: batch: List
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return states, actions, rewards, dones, next_states


def fanin_init(size, fanin: float, device: torch.device = torch.device("cpu")):
    """
    Init weights
    :param size: tensor size
    :param fanin:
    :return:
    """
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v).to(device)


def printAction(action, window):
    sum_list = []
    i = 0
    while i < len(action):
        sum_list.append(sum(action[i: i + window: 1]) / window)
        i = i + window
    plt.plot(sum_list)
    plt.show()


def readDataset(path):
    data = []
    dataset = csv.reader(open(path, encoding='utf_8_sig'), delimiter=',')
    for rol in dataset:
        data.append(rol)
    data = data[1:len(data):1]
    for i in range(len(data)):
        data[i][0] = literal_eval(data[i][0])
        data[i][1] = literal_eval(data[i][1])
        data[i][3] = literal_eval(data[i][3])
        data[i][2] = float(data[i][2])
    return data


def formatnum(x, pos):
    return '$%.1f$x$10^{4}$' % (x / 10000)


def plot_action_distribution(action_list: List, num_nodes: int):
    """
    Plot the distribution of actions
    :param action_list: list of actions
    :param number of nodes
    :return: nothing, just plot
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')

    _x = np.arange(num_nodes)
    _y = np.arange(num_nodes)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    data = np.zeros(shape=(num_nodes*num_nodes))
    bottom = np.zeros_like(data)
    width = depth = 1
    for item in action_list:
        data[item[0]*num_nodes + item[1]] += 1

    ax1.bar3d(x, y, bottom, width, depth, data, shade=True)
    plt.show()

def plotActionTrace(action_trace):
    for key in action_trace.keys():
        plt.plot(action_trace[key], label=str(int(key)))
    plt.xlabel("Iterations")
    plt.ylabel("Action")
    plt.title("Agent's Output with Time")
    plt.ylim((0, 100000))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def report(model: Model):
    fail_rate = model.calculate_fail_rate()
    real_fail_rate = Monitor.calculate_real_fail_rate()
    throughput = model.calculate_throughput()
    service_time = model.calculate_service_time()
    total_reward = model.calculate_total_reward()
    accept_rate = model.calculate_accept_rate()
    # server_rate = model.calculate_server_occupied_rate()
    # link_rate = model.calculate_link_occupied_rate()

    print("fail rate: ", fail_rate)
    print("real fail rate: ", real_fail_rate)
    print("throughput: ", throughput)
    print("service time: ", service_time)
    print("total reward: ", total_reward)
    print("accept rate: ", accept_rate)
    # print("server rate: ", server_rate)
    # print("link rate: ", link_rate)
    return fail_rate, real_fail_rate, throughput, service_time, total_reward, accept_rate


def main():
    pass


if __name__ == '__main__':
    main()
