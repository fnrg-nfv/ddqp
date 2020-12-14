import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import collections
from sfcbased.model import *


class Action:
    pass


class Environment:
    def get_reward(self, model: Model, sfc_index: int):
        return 0

    def get_state(self, model: Model, sfc_index: int):
        return (), False


class NormalEnvironment(Environment):
    def get_reward(self, model: Model, sfc_index: int):
        return 0

    def get_state(self, model: Model, sfc_index: int):
        return (), False


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    """
    experience buffer class
    """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        """
        append an experience item

        :param experience: experience item
        :return: None
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        sample a batch from this buffer

        :param batch_size: sample size
        :return: ()
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return states, actions, rewards, dones, next_states


def fanin_init(size, fanin: float, device: torch.device = torch.device("cpu")):
    """
    init weights

    :param size: tensor size
    :param fanin: range
    :param device: computing device
    :return: torch.Tensor
    """
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v).to(device)


def plot_action_distribution(action_list: List, num_nodes: int):
    """
    plot the distribution of actions

    :param action_list: list of actions
    :param num_nodes: number of nodes
    :return: None
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


def report(model: Model):
    """
    report stats

    :param model: model
    :return: ()
    """
    fail_rate = model.calculate_fail_rate()
    real_fail_rate = Monitor.calculate_real_fail_rate()
    throughput = model.calculate_throughput()
    service_availability = model.calculate_service_availability()
    total_reward = model.calculate_total_reward()
    accept_rate = model.calculate_accept_rate()
    accept_num = model.calculate_accepted_number()
    place_num = model.calculate_place_num()
    place_cdf = model.calculate_place_cdf(num=9)
    # server_rate = model.calculate_server_occupied_rate()
    # link_rate = model.calculate_link_occupied_rate()

    print("fail rate: ", fail_rate)
    print("real fail rate: ", real_fail_rate)
    print("throughput: ", throughput)
    print("service availability: ", service_availability)
    print("total reward: ", total_reward)
    print("accept num: ", accept_num)
    print("place num: ", place_num)
    print("accept rate: ", accept_rate)
    print("place cdf: ", place_cdf)
    # print("server rate: ", server_rate)
    # print("link rate: ", link_rate)
    return fail_rate, real_fail_rate, throughput, service_availability, total_reward, accept_num, place_num, accept_rate, place_cdf


def main():
    pass


if __name__ == '__main__':
    main()
