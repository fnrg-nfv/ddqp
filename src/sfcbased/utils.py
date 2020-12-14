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


class PrioritizedExperienceBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, alpha):
        self.e = 0.01
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def __len__(self):
        return self.tree.n_entries

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.alpha

    def append(self, error: float, sample: Experience):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n: int):
        batch = []
        idxes = []
        segment = self.tree.total() / n
        priorities = []

        # annealing
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            random_val = random.uniform(a, b)
            idx, p, data = self.tree.get(random_val)
            priorities.append(p)
            batch.append([data.state, data.action, data.reward, data.done, data.new_state])
            idxes.append(idx)

        # priority normalization
        sampling_probabilities = np.array(priorities) / self.tree.total()

        # importance sampling weight
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return map(list, zip(*batch)), idxes, is_weight

    def set_priorities(self, indices, errors):
        for i, e in zip(indices, errors):
            p = self._get_priority(e)
            self.tree.update(i, p)


# a binary tree data structure where the parent’s value is the sum of its children
class SumTree(object):
    def __init__(self, capacity: int):
        self.write = 0
        self.capacity = capacity # leaf nodes
        self.tree = np.zeros(2 * capacity - 1) # tree size
        self.data = np.zeros(capacity, dtype=Experience) # corresponding data
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """
        propagate change to the whole tree
        :param idx: index
        :param change: change
        :return: None
        """
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, val: float):
        """
        find sample based on value
        :param idx: idx of tree
        :param val: value
        :return: idx on tree leaf
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if val <= self.tree[left]:
            return self._retrieve(left, val)
        else:
            return self._retrieve(right, val - self.tree[left])

    # sum of all priorities
    def total(self):
        return self.tree[0]

    def add(self, p: float, data: Experience):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx: int, p: float):
        """
        update priority of certain tree node
        :param idx: idx on tree
        :param p: priority
        :return: None
        """
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample from random value
    def get(self, random_val: float) -> (int, float, Experience):
        idx = self._retrieve(0, random_val)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]


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

    # print("fail rate: ", fail_rate)
    # print("real fail rate: ", real_fail_rate)
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
