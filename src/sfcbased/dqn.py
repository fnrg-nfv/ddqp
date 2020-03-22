import torch.nn as nn
from sfcbased.utils import *
from sfcbased.model import *


class DQN(nn.Module):
    def __init__(self, state_len: int, action_len: int, tgt: bool, device: torch.device):
        super(DQN, self).__init__()
        self.tgt = tgt
        self.action_len = action_len
        self.device = device
        self.state_len = state_len
        self.LeakyReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()
        self.ELU = nn.ELU()
        self.Sig = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.BNs = nn.ModuleList()
        self.bn1 = nn.BatchNorm1d(state_len)
        self.num = 512
        self.bn2 = nn.BatchNorm1d(self.num)
        self.bn3 = nn.BatchNorm1d(self.num)

        self.BNs.append(nn.BatchNorm1d(num_features=self.state_len))
        self.fc1 = nn.Linear(in_features=self.state_len, out_features=self.num)
        self.BNs.append(nn.BatchNorm1d(num_features=self.num))
        self.fc2 = nn.Linear(in_features=self.num, out_features=self.num)
        self.BNs.append(nn.BatchNorm1d(num_features=self.num))
        self.fc3 = nn.Linear(in_features=self.num, out_features=self.num)
        self.fc4 = nn.Linear(in_features=self.num, out_features=self.num)
        self.fc5 = nn.Linear(in_features=self.num, out_features=self.num)
        self.fc6 = nn.Linear(in_features=self.num, out_features=self.num)
        self.fc7 = nn.Linear(in_features=self.num, out_features=self.action_len)

        self.init_weights(3e9)

    def init_weights(self, init_w: float):
        for bn in self.BNs:
            bn.weight.data = fanin_init(bn.weight.data.size(), init_w, device=self.device)
            bn.bias.data = fanin_init(bn.bias.data.size(), init_w, device=self.device)
            bn.running_mean.data = fanin_init(bn.running_mean.data.size(), init_w, device=self.device)
            bn.running_var.data = fanin_init(bn.running_var.data.size(), init_w, device=self.device)

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size(), init_w, device=self.device)
        self.fc1.bias.data = fanin_init(self.fc1.bias.data.size(), init_w, device=self.device)

        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size(), init_w, device=self.device)
        self.fc2.bias.data = fanin_init(self.fc2.bias.data.size(), init_w, device=self.device)

        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size(), init_w, device=self.device)
        self.fc3.bias.data = fanin_init(self.fc3.bias.data.size(), init_w, device=self.device)

        self.fc4.weight.data = fanin_init(self.fc4.weight.data.size(), init_w, device=self.device)
        self.fc4.bias.data = fanin_init(self.fc4.bias.data.size(), init_w, device=self.device)

        self.fc5.weight.data = fanin_init(self.fc5.weight.data.size(), init_w, device=self.device)
        self.fc5.bias.data = fanin_init(self.fc5.bias.data.size(), init_w, device=self.device)

        self.fc6.weight.data = fanin_init(self.fc6.weight.data.size(), init_w, device=self.device)
        self.fc6.bias.data = fanin_init(self.fc6.bias.data.size(), init_w, device=self.device)

        self.fc7.weight.data = fanin_init(self.fc7.weight.data.size(), init_w, device=self.device)
        self.fc7.bias.data = fanin_init(self.fc7.bias.data.size(), init_w, device=self.device)

    def forward(self, x: torch.Tensor):
        x.to(device=self.device)

        # x = self.bn1(x)

        # x = self.BNs[0](x)
        x = self.fc1(x)
        x = self.LeakyReLU(x)

        # # x = self.BNs[1](x)
        # x = self.bn2(x)
        x = self.fc2(x)
        x = self.LeakyReLU(x)
        # #
        # # # x = self.BNs[2](x)
        x = self.fc3(x)
        x = self.LeakyReLU(x)
        # # # print("output: ", x)
        #
        # x = self.fc4(x)
        # x = self.LeakyReLU(x)
        # #
        # x = self.fc5(x)
        # x = self.LeakyReLU(x)
        #
        # x = self.fc6(x)
        # x = self.LeakyReLU(x)
        # x = self.bn3(x)
        x = self.fc7(x)
        return x


class DQNDecisionMaker(DecisionMaker):
    def __init__(self, net: DQN, tgt_net: DQN, buffer: ExperienceBuffer, action_space: List, gamma: float,
                 epsilon_start: float, epsilon: float, epsilon_final: float, epsilon_decay: float, model: Model,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.net = net
        self.tgt_net = tgt_net
        self.buffer = buffer
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.gamma = gamma
        self.idx = 0
        self.nodes_num = len(model.topo.nodes)
        self.forbidden_action_index = [1 if i == j else 0 for i in range(self.nodes_num) for j in range(self.nodes_num)]
        self.forbidden_action_index_tensor = torch.tensor([self.forbidden_action_index], device=self.device, dtype=torch.bool)  # forbid the same placement

    def generate_decision(self, model: Model, sfc_index: int, state: List, test_env: TestEnv):
        """
        generate decision for deploying

        :param model: model
        :param sfc_index: current sfc index
        :param state: make decision according to this state
        :param test_env: test environment
        :return: Decision decision
        """
        if self.net.tgt: # when target DQN is running
            # action_indexs = self.narrow_action_index_set_for_tgt(model, sfc_index, test_env)
            action_indexs = []
            if len(action_indexs) != 0:
                action_indexs = torch.tensor(action_indexs, device=self.device)
            state_a = np.array([state], copy=False)  # make state vector become a state matrix
            state_v = torch.tensor(state_a, dtype=torch.float, device=self.device)  # transfer to tensor class
            self.net.eval()
            q_vals_v = self.net(state_v)  # input to network, and get output
            q_vals_v[self.forbidden_action_index_tensor] = -999
            q_vals_v = torch.index_select(q_vals_v, dim=1, index=action_indexs) if len(
                action_indexs) != 0 else q_vals_v  # select columns
            _, act_v = torch.max(q_vals_v, dim=1)  # get the max index
            action_index = action_indexs[int(act_v.item())] if len(action_indexs) != 0 else act_v.item()
        else: # when sample DQN is running
            # action_indexs = self.narrow_action_index_set_for_tgt(model, sfc_index, test_env)
            action_indexs = []
            if len(action_indexs) != 0:
                action_indexs = torch.tensor(action_indexs, device=self.device)
            if np.random.random() < self.epsilon:
                if len(action_indexs) != 0:
                    action = action_indexs[random.randint(0, len(action_indexs) - 1)]
                else:
                    action = random.randint(0, len(self.action_space) - 1)
                    while self.forbidden_action_index[action] == 1:
                        action = random.randint(0, len(self.action_space) - 1)
                action_index = action
            else:
                state_a = np.array([state], copy=False)  # make state vector become a state matrix
                state_v = torch.tensor(state_a, dtype=torch.float, device=self.device)  # transfer to tensor class
                self.net.eval()
                q_vals_v = self.net(state_v)  # input to network, and get output
                q_vals_v[self.forbidden_action_index_tensor] = -999
                q_vals_v = torch.index_select(q_vals_v, dim=1, index=action_indexs) if len(
                    action_indexs) != 0 else q_vals_v  # select columns
                _, act_v = torch.max(q_vals_v, dim=1)  # get the max index
                action_index = action_indexs[int(act_v.item())] if len(action_indexs) != 0 else act_v.item()
        action = self.action_space[action_index]
        decision = Decision()
        decision.active_server = action[0]
        decision.standby_server = action[1]
        self.epsilon = max(self.epsilon_final, self.epsilon_start - self.idx / self.epsilon_decay)
        self.idx += 1
        return decision


class DQNAction(Action):
    def __init__(self, active: int, standby: int):
        super().__init__()
        self.active = active
        self.standby = standby

    def get_action(self):
        """
        get an action tuple

        :return: (int, int) action tuple
        """
        return self.active, self.standby

    def action2index(self, nodes_number: int):
        """
        transfer self to action index

        :param nodes_number: int
        :return: int action index
        """
        return self.active * nodes_number + self.standby


def calc_loss(batch, net, tgt_net, gamma: float, nodes_number: int, double: bool, device: torch.device):
    """
    calculate loss based on batch, net and target net

    feed the states and actions in batch into sample net to get Q-value, then feed the next states in batch and
    next actions into target net to get target Q-value, based on "Q * gamma + rewards_v" and "new Q" to calculate loss

    :param batch: batch
    :param net: sample net
    :param tgt_net: target net
    :param gamma: gamma
    :param nodes_number: nodes number
    :param double: if DDQN or not
    :param device: device
    :return: nn.MSELoss loss
    """
    states, actions, rewards, dones, next_states = batch

    # transform each action to action index
    actions = [DQNAction(action[0], action[1]).action2index(nodes_number) for action in actions]

    states_v = torch.tensor(states, dtype=torch.float).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float).to(device)
    actions_v = torch.tensor(actions, dtype=torch.long).to(device)
    rewards_v = torch.tensor(rewards, dtype=torch.float).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

    # action is a list with one dimension, we should use unsqueeze() to span it
    state_action_values = net(states_v).to(device)
    state_action_values = state_action_values.gather(1, actions_v.unsqueeze(-1))
    state_action_values = state_action_values.squeeze(-1)

    if double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]

    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.to(device).detach()

    expected_state_action_values = next_state_values * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


class DQNEnvironment(Environment):
    def __init__(self):
        super().__init__()

    def get_reward(self, model: Model, sfc_index: int):
        """
        get reward

        :param model: model
        :param sfc_index: sfc index
        :return: double reward
        """
        if model.sfc_list[sfc_index].state == State.Failed:
            return -1
        if model.sfc_list[sfc_index].state == State.Normal:
            return 1

    def get_state(self, model: Model, sfc_index: int):
        """
        get the state of environment

        mainly has two parts:
        1. state of topology
        2. state of sfc

        :param model: model
        :param sfc_index: sfc index
        :return: ([double], bool) state and if it is terminal state
        """
        state = []
        node_len = len(model.topo.nodes)

        # first part: topo state
        # 1. node state
        max_v = max(node[1]['computing_resource'] for node in model.topo.nodes(data=True))
        max_f = max(node[1]['fail_rate'] for node in model.topo.nodes(data=True))

        for node in model.topo.nodes(data=True):
            state.append(node[1]['fail_rate'] / max_f)
            state.append((node[1]['computing_resource'] - node[1]['active']) / max_v)
            if node[1]['reserved'] == float('-inf'):
                state.append(0)
            else:
                state.append(node[1]['reserved'] / max_v)

        # 2. edge state
        max_e = max(edge[2]['bandwidth'] for edge in model.topo.edges(data=True))
        max_l = max(edge[2]['latency'] for edge in model.topo.edges(data=True))

        for edge in model.topo.edges(data=True):
            state.append(edge[2]['latency'] / max_l)
            state.append((edge[2]['bandwidth'] - edge[2]['active']) / max_e)
            if edge[2]['reserved'] == float('-inf'):
                state.append(0)
            else:
                state.append(edge[2]['reserved'] / max_e)

        # second part
        sfc = model.sfc_list[sfc_index] if sfc_index < len(model.sfc_list) else model.sfc_list[sfc_index - 1]
        state.append(sfc.computing_resource / max_v)
        state.append(sfc.tp / max_e)
        state.append(sfc.latency / max_l)
        state.append(sfc.update_tp / max_e)
        state.append(sfc.process_latency / max_l)
        state.append(sfc.s)
        state.append(sfc.d)
        return tuple(state), False
