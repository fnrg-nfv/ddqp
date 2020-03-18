import torch.nn as nn
from sfcbased.utils import *
from sfcbased.model import *


@unique
class Space(Enum):
    Unlimit = 0


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
    """
    This class is denoted as a decision maker used reinforcement learning
    """
    def verify_standby_for_tgt(self, model: Model, cur_sfc_index: int, active_server_index: int, cur_server_index: int,
                       test_env: TestEnv):
        """
        Verify if current stand-by sfc can be put on current server based on following three principles
        1. if the remaining computing resource is still enough for this sfc
        2. if available paths for updating still exist
        Both these three principles are met can return true, else false
        When the active instance is deployed, the topology will change and some constraints may not be met, but this is just a really small case so that we don't have to consider it.
        :param test_env:
        :param model: model
        :param cur_sfc_index: current sfc index
        :param active_server_index: active server index
        :param cur_server_index: current server index
        :return: true or false
        """
        assert test_env != TestEnv.NoBackup
        # principle 1
        if test_env == TestEnv.Aggressive:
            if model.topo.nodes[cur_server_index]["computing_resource"] < model.sfc_list[
                cur_sfc_index].computing_resource:
                return False
        if test_env == TestEnv.Normal or test_env == TestEnv.MaxReservation:
            if model.topo.nodes[cur_server_index]["computing_resource"] - model.topo.nodes[cur_server_index]["active"] < \
                    model.sfc_list[cur_sfc_index].computing_resource:
                return False
        if test_env == TestEnv.FullyReservation:
            if model.topo.nodes[cur_server_index]["computing_resource"] - model.topo.nodes[cur_server_index]["active"] - \
                    model.topo.nodes[cur_server_index]["reserved"] < model.sfc_list[cur_sfc_index].computing_resource:
                return False

        # principle 2
        for path in nx.all_shortest_paths(model.topo, active_server_index, cur_server_index):
            if self.is_path_throughput_met(model, path, model.sfc_list[cur_sfc_index].update_tp, SFCType.Active,
                                           test_env):
                return True
        return False


    def verify_standby(self, model: Model, cur_sfc_index: int, active_server_index: int, cur_server_index: int,
                       test_env: TestEnv):
        """
        Verify if current stand-by sfc can be put on current server based on following three principles
        1. if the remaining computing resource is still enough for this sfc
        2. if available paths for updating still exist
        3. if available paths still exist
        Both these three principles are met can return true, else false
        When the active instance is deployed, the topology will change and some constraints may not be met, but this is just a really small case so that we don't have to consider it.
        :param test_env:
        :param model: model
        :param cur_sfc_index: current sfc index
        :param active_server_index: active server index
        :param cur_server_index: current server index
        :return: true or false
        """
        assert test_env != TestEnv.NoBackup
        # principle 1
        if test_env == TestEnv.Aggressive:
            if model.topo.nodes[cur_server_index]["computing_resource"] < model.sfc_list[
                cur_sfc_index].computing_resource:
                return False
        if test_env == TestEnv.Normal or test_env == TestEnv.MaxReservation:
            if model.topo.nodes[cur_server_index]["computing_resource"] - model.topo.nodes[cur_server_index]["active"] < \
                    model.sfc_list[cur_sfc_index].computing_resource:
                return False
        if test_env == TestEnv.FullyReservation:
            if model.topo.nodes[cur_server_index]["computing_resource"] - model.topo.nodes[cur_server_index]["active"] - \
                    model.topo.nodes[cur_server_index]["reserved"] < model.sfc_list[cur_sfc_index].computing_resource:
                return False

        # principle 2
        principle2 = False
        for path in nx.all_shortest_paths(model.topo, active_server_index, cur_server_index):
            if self.is_path_throughput_met(model, path, model.sfc_list[cur_sfc_index].update_tp, SFCType.Active,
                                           test_env):
                principle2 = True
                break
        if not principle2:
            return False

        # principle 3
        for path_s2c in nx.all_shortest_paths(model.topo, model.sfc_list[cur_sfc_index].s, cur_server_index):
            if self.is_path_throughput_met(model, path_s2c, model.sfc_list[cur_sfc_index].tp, SFCType.Standby,
                                           test_env):
                for path_c2d in nx.all_shortest_paths(model.topo, cur_server_index, model.sfc_list[cur_sfc_index].d):
                    if self.is_path_latency_met(model, path_s2c, path_c2d,
                                                model.sfc_list[cur_sfc_index].latency - model.sfc_list[
                                                    cur_sfc_index].process_latency) and self.is_path_throughput_met(
                        model,
                        path_c2d,
                        model.sfc_list[
                            cur_sfc_index].tp,
                        SFCType.Standby, test_env):
                        return True
        return False

    def narrow_action_index_set(self, model: Model, cur_sfc_index: int, test_env: TestEnv):
        """
        Used to narrow available decision set
        :param test_env: test env
        :param model: model
        :param cur_sfc_index: cur processing sfc index
        :return: action index sets
        """
        action_index_set = []
        for i in range(len(model.topo.nodes)):
            if not self.verify_active(model, cur_sfc_index, i, test_env):
                continue
            if test_env == TestEnv.NoBackup:
                action_index_set.append(i)
                continue
            for j in range(len(model.topo.nodes)):
                if i != j and self.verify_standby(model, cur_sfc_index, i, j, test_env):
                    action_index_set.append(i * len(model.topo.nodes) + j)
        return action_index_set

    def narrow_action_index_set_for_tgt(self, model: Model, cur_sfc_index: int, test_env: TestEnv):
        """
        Used to narrow available decision set
        :param test_env: test env
        :param model: model
        :param cur_sfc_index: cur processing sfc index
        :return: action index sets
        """
        action_index_set = []
        for i in range(len(model.topo.nodes)):
            if not self.verify_active(model, cur_sfc_index, i, test_env):
                continue
            if test_env == TestEnv.NoBackup:
                action_index_set.append(i)
                continue
            for j in range(len(model.topo.nodes)):
                if i != j and self.verify_standby_for_tgt(model, cur_sfc_index, i, j, test_env):
                    action_index_set.append(i * len(model.topo.nodes) + j)
        return action_index_set

    def __init__(self, net: DQN, tgt_net: DQN, buffer: ExperienceBuffer, action_space: List, gamma: float, epsilon_start: float, epsilon: float, epsilon_final: float, epsilon_decay: float, model: Model, device: torch.device = torch.device("cpu")):
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
        self.forbidden_action_index_tensor = torch.tensor([[1 if i == j else 0 for i in range(self.nodes_num) for j in range(self.nodes_num)]], device=self.device, dtype=torch.bool)
        self.forbidden_action_index = [1 if i == j else 0 for i in range(self.nodes_num) for j in range(self.nodes_num)]


    def generate_decision(self, model: Model, cur_sfc_index: int, state: List, test_env: TestEnv):
        if self.net.tgt:
            # action_indexs = self.narrow_action_index_set_for_tgt(model, cur_sfc_index, test_env)
            action_indexs = []
            if len(action_indexs) != 0:
                action_indexs = torch.tensor(action_indexs, device=self.device)
            state_a = np.array([state], copy=False)  # make state vector become a state matrix
            state_v = torch.tensor(state_a, dtype=torch.float, device=self.device)  # transfer to tensor class
            self.net.eval()
            q_vals_v = self.net(state_v)  # input to network, and get output
            q_vals_v[self.forbidden_action_index_tensor] = -999
            q_vals_v = torch.index_select(q_vals_v, dim=1, index=action_indexs) if len(action_indexs) != 0 else q_vals_v # select columns
            _, act_v = torch.max(q_vals_v, dim=1)  # get the max index
            action_index = action_indexs[int(act_v.item())] if len(action_indexs) != 0 else act_v.item()
        else:
            # action_indexs = self.narrow_action_index_set_for_tgt(model, cur_sfc_index, test_env)
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
                q_vals_v = torch.index_select(q_vals_v, dim=1, index=action_indexs) if len(action_indexs) != 0 else q_vals_v # select columns
                _, act_v = torch.max(q_vals_v, dim=1)  # get the max index
                action_index = action_indexs[int(act_v.item())] if len(action_indexs) != 0 else act_v.item()
        action = self.action_space[action_index]
        # print(action)
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
        return [self.active, self.standby]

    def action2index(self, action_space: List):
        for i in range(len(action_space)):
            if action_space[i][0] == self.active and action_space[i][1] == self.standby:
                return i
        raise RuntimeError("The action space doesn't contain this action")


def calc_loss(batch, net, tgt_net, gamma: float, action_space: List, double: bool, device: torch.device):
    states, actions, rewards, dones, next_states = batch

    # transform each action to index(real action)
    actions = [DQNAction(action[0], action[1]).action2index(action_space) for action in actions]

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

    def get_reward(self, model: Model, sfc_index: int, decision: Decision, test_env: TestEnv):
        if model.sfc_list[sfc_index].state == State.Failed:
            return -1
        if model.sfc_list[sfc_index].state == State.Normal:
            return 1

        # reward -= model.topo.nodes(data=True)[decision.standby_server]["fail_rate"]
        # reward = reward - model.topo.nodes(data=True)[decision.standby_server]["fail_rate"]

    def get_state(self, model: Model, sfc_index: int):
        """
        Get the state of current network.
        :param model: model
        :param sfc_indexs: sfc indexs
        :param process_capacity: process capacity
        :return: state vector, done
        """
        state = []
        node_len = len(model.topo.nodes)

        # first part: topo state
        # 1. node state
        max_v = 0
        for node in model.topo.nodes(data=True):
            if node[1]['computing_resource'] > max_v:
                max_v = node[1]['computing_resource']
        max_f = 0
        for node in model.topo.nodes(data=True):
            if node[1]['fail_rate'] > max_f:
                max_f = node[1]['fail_rate']
        for node in model.topo.nodes(data=True):
            state.append(node[1]['fail_rate'] / max_f)
            state.append((node[1]['computing_resource'] - node[1]['active'])/ max_v)
            if node[1]['reserved'] == float('-inf'):
                state.append(0)
            else:
                state.append(node[1]['reserved'] / max_v)

        # 2. edge state
        max_e = 0
        for edge in model.topo.edges(data=True):
            if edge[2]['bandwidth'] > max_e:
                max_e = edge[2]['bandwidth']
        max_l = 0
        for edge in model.topo.edges(data=True):
            if edge[2]['latency'] > max_l:
                max_l = edge[2]['latency']
        for edge in model.topo.edges(data=True):
            state.append(edge[2]['latency'] / max_l)
            state.append((edge[2]['bandwidth'] - edge[2]['active']) / max_e)
            if edge[2]['reserved'] == float('-inf'):
                state.append(0)
            else:
                state.append(edge[2]['reserved'] / max_e)

        # the sfcs located in this time slot state
        sfc = model.sfc_list[sfc_index] if sfc_index < len(model.sfc_list) else model.sfc_list[sfc_index - 1]
        state.append(sfc.computing_resource / max_v)
        state.append(sfc.tp / max_e)
        state.append(sfc.latency / max_l)
        state.append(sfc.update_tp / max_e)
        state.append(sfc.process_latency / max_l)
        state.append(sfc.s)
        state.append(sfc.d)
        return state, False

        #second part
        #current sfc hasn't been deployed
        # if sfc_index == len(model.sfc_list) - 1 or model.sfc_list[sfc_index].state == State.Undeployed:
        #     sfc = model.sfc_list[sfc_index]
        #     state.append(sfc.computing_resource)
        #     state.append(sfc.tp)
        #     state.append(sfc.latency)
        #     state.append(sfc.update_tp)
        #     state.append(sfc.process_latency)
        #     state.append(sfc.s)
        #     state.append(sfc.d)
        #
        # #current sfc has been deployed
        # elif model.sfc_list[sfc_index].state == State.Normal or model.sfc_list[sfc_index].state == State.Failed:
        #     sfc = model.sfc_list[sfc_index + 1]
        #     state.append(sfc.computing_resource)
        #     state.append(sfc.tp)
        #     state.append(sfc.latency)
        #     state.append(sfc.update_tp)
        #     state.append(sfc.process_latency)
        #     state.append(sfc.s)
        #     state.append(sfc.d)
