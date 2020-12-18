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

    def append_sample(self, exp: Experience, gamma: float):
        if isinstance(self.buffer, ExperienceBuffer):
            self.buffer.append(exp)
        elif isinstance(self.buffer, PrioritizedExperienceBuffer):
            states_v = torch.tensor([exp.state], dtype=torch.float).to(self.device)
            next_states_v = torch.tensor([exp.new_state], dtype=torch.float).to(self.device)
            actions_v = torch.tensor([exp.action], dtype=torch.long).to(self.device)
            rewards_v = torch.tensor([exp.reward], dtype=torch.float).to(self.device)
            done_mask = torch.tensor([exp.done], dtype=torch.bool).to(self.device)
            state_action_values = self.net(states_v).to(self.device)
            state_action_values = state_action_values.gather(1, actions_v.unsqueeze(-1))
            state_action_values = state_action_values.squeeze(-1)
            next_state_actions = self.net(next_states_v).max(1)[1]
            next_state_values = self.tgt_net(next_states_v).gather(
                1, next_state_actions.unsqueeze(-1)).squeeze(-1)
            expected_state_action_values = next_state_values * gamma * done_mask + rewards_v
            td_errors = expected_state_action_values - state_action_values
            td_error = abs(td_errors.sum())
            self.buffer.append(float(td_error), exp)

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


class BranchingQNetwork(nn.Module):
    def __init__(self, state_len: int, dimensions: int, actions_per_dimension: int, is_tgt: bool, is_fc: bool,
                 device: torch.device):
        super(BranchingQNetwork, self).__init__()
        self.is_tgt = is_tgt
        self.is_fc = is_fc
        self.device = device
        self.num = 256  # node nums

        # scales
        self.state_len = state_len
        self.dimensions = dimensions
        self.actions_per_dimension = actions_per_dimension

        ## self.before_bn = nn.BatchNorm1d(self.state_len)
        self.shared_model = nn.Sequential(nn.Linear(self.state_len, self.num),
                                          nn.LeakyReLU(),
                                          ## nn.BatchNorm1d(self.num),
                                          nn.Linear(self.num, self.num),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.num, self.num),
                                          nn.LeakyReLU()).to(self.device)

        self.value_head = nn.Sequential(nn.Linear(self.num, self.num),
                                        nn.LeakyReLU(),
                                        nn.Linear(self.num, self.num),
                                        nn.LeakyReLU(),
                                        nn.Linear(self.num, 1)).to(self.device)

        self.shared_model[0].weight.data.uniform_(-0.0001, 0.0001)
        self.shared_model[2].weight.data.uniform_(-0.0001, 0.0001)
        self.shared_model[4].weight.data.uniform_(-0.0001, 0.0001)
        self.value_head[0].weight.data.uniform_(-0.0001, 0.0001)
        self.value_head[2].weight.data.uniform_(-0.0001, 0.0001)
        self.value_head[4].weight.data.uniform_(-0.0001, 0.0001)

        if self.is_fc:
            self.adv_heads = nn.Sequential(nn.Linear(self.num, self.num * self.dimensions),
                                           nn.LeakyReLU(),
                                           nn.Linear(self.num * self.dimensions, self.num * self.dimensions),
                                           nn.LeakyReLU(),
                                           nn.Linear(self.num * self.dimensions,
                                                     self.actions_per_dimension * self.dimensions)).to(self.device)
            self.adv_heads[0].weight.data.uniform_(-0.0001, 0.0001)
            self.adv_heads[2].weight.data.uniform_(-0.0001, 0.0001)
            self.adv_heads[4].weight.data.uniform_(-0.0001, 0.0001)
        else:
            self.adv_heads = nn.ModuleList(
                [nn.Sequential(nn.Linear(self.num, self.num),
                               nn.LeakyReLU(),
                               nn.Linear(self.num, self.num),
                               nn.LeakyReLU(),
                               nn.Linear(self.num, self.actions_per_dimension)) for i in range(self.dimensions)]).to(self.device)
            for layer in self.adv_heads:
                layer[0].weight.data.uniform_(-0.0001, 0.0001)
                layer[2].weight.data.uniform_(-0.0001, 0.0001)
                layer[4].weight.data.uniform_(-0.0001, 0.0001)

    def forward(self, x: torch.Tensor):
        x.to(device=self.device)

        ## out = self.after_bn(self.model(self.before_bn(x)))
        out = self.shared_model(x)

        value = self.value_head(out)

        if self.is_fc:
            # [batch_size, dimension * act_in_each_dimension]
            advs = self.adv_heads(out)
            # [batch_size, dimension * act_in_each_dimension]
            # [batch_size, dimension, act_in_each_dimension]
            advs = advs.reshape((len(advs), self.dimensions, self.actions_per_dimension))
        else:
            # [dimension, batch_size, act_in_each_dimension] to
            # [batch_size, dimension, act_in_each_dimension]
            advs = torch.stack([l(out) for l in self.adv_heads], dim=1)

        # [batch_size, dimension, act_in_each_dimension] to
        # [batch_size, dimension, 1]
        mean = advs.mean(2, keepdim=True)

        # [batch_size, 1] to
        # [batch_size, 1, 1] to
        # [batch_size, dimension, act_in_each_dimension]
        q_val = value.unsqueeze(2) + advs - mean

        # [batch_size, dimension, act_in_each_dimension]
        return q_val


class BranchingDecisionMaker(DecisionMaker):
    def __init__(self, net: DQN, tgt_net: DQN, buffer, gamma: float,
                 epsilon_start: float, epsilon: float, epsilon_final: float, epsilon_decay: float, model: Model,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.net = net
        self.tgt_net = tgt_net
        self.buffer = buffer
        self.epsilon = epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.gamma = gamma
        self.idx = 0

    def append_sample(self, exp: Experience, gamma: float):
        if isinstance(self.buffer, ExperienceBuffer):
            self.buffer.append(exp)
        elif isinstance(self.buffer, PrioritizedExperienceBuffer):
            states = torch.tensor([exp.state]).float().to(self.device)
            actions = torch.tensor([exp.action]).long().reshape(states.shape[0], -1, 1).to(self.device)
            rewards = torch.tensor([exp.reward]).float().reshape(-1, 1).to(self.device)
            next_states = torch.tensor([exp.new_state]).float().to(self.device)
            masks = torch.tensor([exp.done]).float().reshape(-1, 1).to(self.device)
            current_q_vals = self.net(states).gather(2, actions).squeeze(-1)
            argmax = torch.argmax(self.net(next_states), dim=2)
            next_q_vals = self.tgt_net(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)
            next_q_vals = next_q_vals.mean(1, keepdim=True).expand(next_q_vals.shape)
            target_q_vals = rewards + next_q_vals * gamma * masks
            dim_td_errors = target_q_vals - current_q_vals
            td_error = torch.abs(dim_td_errors).sum()
            self.buffer.append(float(td_error), exp)

    def generate_decision(self, model: Model, sfc_index: int, state: List, test_env: TestEnv):
        """
        generate decision for deploying

        :param model: model
        :param sfc_index: current sfc index
        :param state: make decision according to this state
        :param test_env: test environment
        :return: Decision decision
        """
        if self.net.is_tgt: # when target DQN is running
            state_a = np.array([state], copy=False)  # make state vector become a state matrix
            state_v = torch.tensor(state_a, dtype=torch.float, device=self.device)  # transfer to tensor class
            self.net.eval()
            q_vals_v = self.net(state_v).squeeze(0)  # input to network, and get output
            action = torch.argmax(q_vals_v, dim=1)
            if action[0] == action[1]:
                q_vals_v[1][action[1]] = float("-inf")
                action = torch.argmax(q_vals_v, dim=1)
            assert action[0] != action[1]
            action = action.tolist()
        else: # when sample DQN is running
            if np.random.random() < self.epsilon:
                action = [0, 0]
                action[0] = random.randint(0, self.net.actions_per_dimension - 1)
                action[1] = random.randint(0, self.net.actions_per_dimension - 1)
                while action[0] == action[1]:
                    action[1] = random.randint(0, self.net.actions_per_dimension - 1)
            else:
                state_a = np.array([state], copy=False)  # make state vector become a state matrix
                state_v = torch.tensor(state_a, dtype=torch.float, device=self.device)  # transfer to tensor class
                self.net.eval()
                q_vals_v = self.net(state_v).squeeze(0)  # input to network, and get output
                action = torch.argmax(q_vals_v, dim=1)
                if action[0] == action[1]:
                    q_vals_v[1][action[1]] = float("-inf")
                    action = torch.argmax(q_vals_v, dim=1)
                assert action[0] != action[1]
                action = action.tolist()
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


def calc_loss_branching_prio(batch, net, tgt_net, gamma: float, indices, importances, buffer, device: torch.device):
    is_global = True
    is_importance = False

    states, actions, rewards, dones, next_states = batch

    # [batch_size, len(global_state)]
    states = torch.tensor(states).float().to(device)

    # [batch_size, dimension] to
    # [batch_size, dimension, 1]
    actions = torch.tensor(actions).long().reshape(states.shape[0], -1, 1).to(device)

    # [batch_size] to
    # [batch_size, 1]
    rewards = torch.tensor(rewards).float().reshape(-1, 1).to(device)

    # print("rewards: ", rewards.tolist())

    # [batch_size, len(global_state)]
    next_states = torch.tensor(next_states).float().to(device)

    # [batch_size] to
    # [batch_size, 1]
    masks = torch.tensor(dones).float().reshape(-1, 1).to(device)

    # [batch_size]
    importances = torch.tensor(importances).float().to(device)

    # only care about the values in specified actions
    # [batch_size, dimension, act_in_each_dimension] to
    # [batch_size, dimension, 1] to
    # [batch_size, dimension]
    current_q_vals = net(states).gather(2, actions).squeeze(-1)

    with torch.no_grad():
        # [batch_size, len(global_state)] to
        # [batch_size, dimension, act_in_each_dimension] to
        # [batch_size, dimension] (batch of action)
        argmax = torch.argmax(net(next_states), dim=2)

        # [batch_size, dimension, act_in_each_dimension] gather [batch_size, dimension, 1] to
        # [batch_size, dimension, 1] to
        # [batch_size, dimension]
        next_q_vals = tgt_net(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)

        if is_global:
            # [batch_size, dimension] to
            # [batch_size, 1] to
            # [batch_size, dimension]
            global_next_q_vals = next_q_vals.mean(1, keepdim=True).expand(next_q_vals.shape)
            next_q_vals = global_next_q_vals

        # [batch_size, dimension]
        target_q_vals = rewards + next_q_vals * gamma * masks

    # [batch_size, dimension]
    dim_td_errors = target_q_vals - current_q_vals

    # according to ABA, a new td error for PER should be specified
    # [batch_size, dimension] to
    # [batch_size]
    td_errors = torch.abs(dim_td_errors).sum(1)

    # update td errors
    buffer.set_priorities(indices, td_errors.tolist())

    # print("td errors: ", td_errors)

    # [batch_size]
    error_for_each_exp = dim_td_errors.pow(2).sum(1) / dim_td_errors.shape[1]

    # according to ABA, the performance of them are the same
    if is_importance:
        return torch.sum(error_for_each_exp * importances)
    else:
        return torch.mean(error_for_each_exp)

    # return nn.MSELoss()(, torch.zeros(1).expand(td_errors.shape))

    # loss = torch.tensor((dim_td_errors.pow(2).sum(1) / dim_td_errors.shape[1] * importances).pow(2).sum() / dim_td_errors.shape[0]).float().to(device)
    # return loss
    # return nn.MSELoss()(current_q_vals, expected_q_vals.expand(current_q_vals.shape))


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
