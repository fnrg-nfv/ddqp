from typing import List
import networkx as nx
from enum import Enum, unique
import random
from abc import abstractmethod


class VirtualException(BaseException):
    def __init__(self, _type, _func):
        BaseException(self)


class BaseObject(object):
    def __repr__(self):
        """
        When function print() is called, this function will determine what to display
        :return: the __str__() result of current instance
        """
        return self.__str__()


@unique
class SolutionType(Enum):
    Classic = 0
    RL = 1


@unique
class BrokenReason(Enum):
    NoReason = 0
    TimeExpired = 1
    StandbyDamage = 2
    StandbyStartFailed = 3
    ActiveDamage = 4  # for NoBackup condition


@unique
class State(Enum):
    Undeployed = 0
    Failed = 1
    Normal = 2
    Backup = 3
    Broken = 4


@unique
class VariableState(Enum):
    Uninitialized = 0


@unique
class TestEnv(Enum):
    NoBackup = 0
    Aggressive = 1
    Normal = 2
    MaxReservation = 3
    FullyReservation = 4


@unique
class SFCType(Enum):
    Active = 0
    Standby = 1


class Decision(BaseObject):
    """
    This class is denoted as a decision
    """

    def __init__(self, active_server: int = VariableState.Uninitialized,
                 standby_server: int = VariableState.Uninitialized):
        """
        Initialization
        :param active_server: server index of active instance
        :param standby_server: server index of standby instance, if not backup, then -1
        """
        self.flag = VariableState.Uninitialized
        self.active_server = active_server
        self.standby_server = standby_server
        self.active_path_s2c = VariableState.Uninitialized
        self.standby_path_s2c = VariableState.Uninitialized
        self.active_path_c2d = VariableState.Uninitialized
        self.standby_path_c2d = VariableState.Uninitialized
        self.update_path = VariableState.Uninitialized

    def set_active_path_s2c(self, path: List):
        self.active_path_s2c = path

    def set_standby_path_s2c(self, path: List):
        self.standby_path_s2c = path

    def set_active_path_c2d(self, path: List):
        self.active_path_c2d = path

    def set_standby_path_c2d(self, path: List):
        self.standby_path_c2d = path

    def set_update_path(self, path: List):
        self.update_path = path


class Monitor(BaseObject):
    """
    Designed for Monitoring the actions of whole system
    """

    action_list = []
    format_logs = []

    @classmethod
    def state_transition(cls, time: int, sfc_index: int, pre_state: State, new_state: State,
                         reason: BrokenReason = BrokenReason.NoReason):
        """
        Handle the state transition condition
        :param time: occur time
        :param sfc_index: sfc index
        :param pre_state: previous state
        :param new_state: new state
        :param reason: the broken reason
        :return: nothing
        """
        if reason == BrokenReason.NoReason:
            cls.log(
                "At time {}, the state of SFC {} changes from {} to {}".format(time, sfc_index, pre_state, new_state))
            cls.format_log([time, sfc_index, pre_state, new_state])
        else:
            cls.log("At time {}, the state of SFC {} changes from {} to {}, for {}".format(time, sfc_index, pre_state,
                                                                                           new_state, reason))
            cls.format_log([time, sfc_index, pre_state, new_state, reason])

    @classmethod
    def log(cls, content: str):
        cls.action_list.append(content)

    @classmethod
    def format_log(cls, content: List):
        cls.format_logs.append(content)

    @classmethod
    def deploy_server(cls, sfc_index: int, server_id: int):
        cls.log("SFC {} deploy on server {}".format(sfc_index, server_id))

    @classmethod
    def active_computing_resource_change(cls, server_id: int, before: int, after: int):
        cls.log("The active computing resource of server {} from {} changes to {}".format(server_id, before, after))

    @classmethod
    def active_bandwidth_change(cls, start: int, destination: int, before: int, after: int):
        cls.log("The active bandwidth of link from {} to {} changes from {} to {}".format(start, destination, before,
                                                                                          after))

    @classmethod
    def reserved_computing_resource_change(cls, server_id: int, before: int, after: int):
        cls.log("The reserved computing resource of server {} from {} changes to {}".format(server_id, before, after))

    @classmethod
    def reserved_bandwidth_change(cls, start: int, destination: int, before: int, after: int):
        cls.log("The reserved bandwidth of link from {} to {} changes from {} to {}".format(start, destination, before,
                                                                                            after))

    @classmethod
    def print_log(cls):
        for item in cls.action_list:
            print(item)

    @classmethod
    def calculate_real_fail_rate(cls):
        fail_num = 0
        success_num = 0
        for item in cls.format_logs:
            if len(item) == 5:
                if item[4] == BrokenReason.StandbyStartFailed:
                    fail_num += 1
            if len(item) == 4:
                if item[2] == State.Normal and item[3] == State.Backup:
                    success_num += 1
        if fail_num == 0:
            return 0
        return fail_num / (fail_num + success_num)

class Instance(BaseObject):
    """
    This class is denoted as an instance.
    """

    def __init__(self, sfc_index: int, is_active: bool):
        self.sfc_index = sfc_index
        self.is_active = is_active

    def __str__(self):
        if self.is_active:
            return "Server {} active instance".format(self.sfc_index)
        else:
            return "Server {} stand-by instance".format(self.sfc_index)


class ACSFC(BaseObject):
    """
    This class is denoted as an active SFC.
    """

    def __init__(self):
        self.server = VariableState.Uninitialized
        self.starttime = VariableState.Uninitialized
        self.downtime = VariableState.Uninitialized
        self.path_s2c = VariableState.Uninitialized
        self.path_c2d = VariableState.Uninitialized


class SBSFC(BaseObject):
    """
    This class is denoted as a stand-by SFC.
    """

    def __init__(self):
        self.server = VariableState.Uninitialized
        self.starttime = VariableState.Uninitialized
        self.downtime = VariableState.Uninitialized
        self.path_s2c = VariableState.Uninitialized
        self.path_c2d = VariableState.Uninitialized


class SFC(BaseObject):
    """
    This class is denoted as a SFC
    """

    def __init__(self, computing_resource: int, tp: int, latency: float, update_tp: int, process_latency: float, s: int,
                 d: int, time: float, TTL: int):
        """
        SFC initialization
        :param computing_resource: computing_resource required
        :param tp: totally throughput required
        :param latency: totally latency required
        :param update_tp: update throughput required
        :param process_latency: latency of processing
        :param s: start server
        :param d: destination server
        :param time: arriving time
        :param TTL: time to live
        """
        self.computing_resource = computing_resource
        self.tp = tp
        self.latency = latency
        self.update_tp = update_tp
        self.process_latency = process_latency
        self.s = s
        self.d = d
        self.time = time
        self.TTL = TTL

        self.state = State.Undeployed
        self.update_path = VariableState.Uninitialized
        self.active_sfc = ACSFC()
        self.standby_sfc = SBSFC()

    def __str__(self):
        """
        Display in console with specified format.
        :return: display string
        """
        return "(computing_resource: {}, throughput: {}, latency: {}, update throughput: {}, process latency: {}, from {}->{}, time: {}, TTL: {})".format(
            self.computing_resource,
            self.tp,
            self.latency, self.update_tp,
            self.process_latency,
            self.s, self.d, self.time, self.TTL)

    def set_state(self, time: int, sfc_index: int, new_state: State, reason: BrokenReason = BrokenReason.NoReason):
        """
        Setting up new state
        :param sfc_index:
        :param reason:
        :param time: occur time
        :param new_state: new state
        :return: nothing
        """

        # Monitor the state transition
        Monitor.state_transition(time, sfc_index, self.state, new_state, reason)

        # set the start time and down time of each sfc instance
        if self.state == State.Undeployed:
            if new_state == State.Failed:
                self.active_sfc.starttime = -1
                self.active_sfc.downtime = -1
                self.standby_sfc.starttime = -1
                self.standby_sfc.downtime = -1
            if new_state == State.Normal:
                self.active_sfc.starttime = self.time
        if self.state == State.Normal:
            self.active_sfc.downtime = time
            if new_state == State.Backup:
                self.standby_sfc.starttime = time
            if new_state == State.Broken:
                self.standby_sfc.starttime = -1
                self.standby_sfc.downtime = -1
        if self.state == State.Backup:
            if new_state == State.Broken:
                if reason == BrokenReason.TimeExpired:
                    self.standby_sfc.downtime = self.time + self.TTL
                else:
                    self.standby_sfc.downtime = time

        self.state = new_state


class Model(BaseObject):
    """
    This class is denoted as the model, a model contains following:
    1. the topology of the whole network
    2. the ordered SFCs need to be deployed
    """

    def __init__(self, topo: nx.Graph, sfc_list: List[SFC]):
        """
        Initialization
        :param topo: network topology
        :param sfc_list: SFCs set
        """
        self.topo = topo
        self.sfc_list = sfc_list

    def __str__(self):
        """
        Display in console with specified format.
        :return: display string
        """
        return "TOPO-nodes:\n{}\nTOPO-edges:\n{}\nSFCs:\n{}".format(self.topo.nodes.data(), self.topo.edges.data(),
                                                                    self.sfc_list)

    def print_start_and_down(self):
        """
        Print out the start time and down time of each instance of each sfc
        :return: nothing
        """
        for i in range(len(self.sfc_list)):
            print(
                "SFC {}:\n   active started at time {} downed at time {}\n   stand-by started at time {} downed at time {}\n".format(
                    i,
                    self.sfc_list[i].active_sfc.starttime,
                    self.sfc_list[i].active_sfc.downtime,
                    self.sfc_list[i].standby_sfc.starttime,
                    self.sfc_list[i].standby_sfc.downtime))

    def calculate_throughput(self):
        """
        Calculate throughput
        :return: throughput
        """
        throughput = 0
        for i in range(len(self.sfc_list)):
            cur_sfc = self.sfc_list[i]
            if cur_sfc.state != State.Failed:
                throughput += cur_sfc.tp
        return throughput

    def calculate_total_reward(self):
        """
        Calculate throughput
        :return: throughput
        """
        total_reward = 0
        for i in range(len(self.sfc_list)):
            cur_sfc = self.sfc_list[i]
            if cur_sfc.state != State.Failed:
                total_reward += 1
        return total_reward

    def calculate_service_time(self):
        """
        Calculate throughput
        :return: throughput
        """
        time = 0
        for i in range(len(self.sfc_list)):
            cur_sfc = self.sfc_list[i]
            if cur_sfc.state == State.Broken:
                time += cur_sfc.active_sfc.downtime - cur_sfc.active_sfc.starttime + cur_sfc.standby_sfc.downtime - cur_sfc.standby_sfc.starttime
        return time

    def calculate_fail_rate(self):
        """
        Calculate fail rate
        :return: fail rate
        """
        real_not_service = 0
        should_not_service = 0
        for i in range(len(self.sfc_list)):
            cur_sfc = self.sfc_list[i]
            if cur_sfc.state == State.Broken:
                should_not_service += cur_sfc.time + cur_sfc.TTL - cur_sfc.active_sfc.downtime
                real_not_service += cur_sfc.time + cur_sfc.TTL - cur_sfc.active_sfc.downtime - (
                        cur_sfc.standby_sfc.downtime - cur_sfc.standby_sfc.starttime)
        if should_not_service == 0:
            return 0
        return real_not_service / should_not_service

    def calculate_accept_rate(self):
        """
        Calculate accept rate
        :return:
        """
        num = 0
        for item in self.sfc_list:
            if item.state == State.Failed:
                num += 1
        return 1 - num / len(self.sfc_list)

    def calculate_server_occupied_rate(self):
        """
        Calculate accept rate
        :return: a list
        """
        server_rate = []
        for node in self.topo.nodes(data=True):
            server = dict()
            server["active"] = node[1]['active'] / node[1]['computing_resource']
            server["reserved"] = node[1]['reserved'] / node[1]['computing_resource']
            server_rate.append(server)
        return server_rate

    def calculate_link_occupied_rate(self):
        """
        Calculate accept rate
        :return: a list
        """
        link_rate = []
        for edge in self.topo.edges(data=True):
            link = dict()
            link["active"] = edge[2]['active'] / edge[2]['bandwidth']
            link["reserved"] = edge[2]['reserved'] / edge[2]['bandwidth']
            if link["active"] > 1:
                print("active: ", edge[2]['active'], edge[2]['bandwidth'])
                print(edge[2]["sbsfcs_s2c"], edge[2]["sbsfcs_c2d"])
            if link["reserved"] > 1:
                print("reserved: ", edge[2]['reserved'], edge[2]['bandwidth'])
                print(edge[2]["sbsfcs_s2c"], edge[2]["sbsfcs_c2d"])
            link_rate.append(link)
        return link_rate

class DecisionMaker(BaseObject):
    """
    The class used to make deploy decision
    """

    def __init__(self):
        super(DecisionMaker, self).__init__()

    def is_path_throughput_met(self, model: Model, path: List, throughput: int, cur_sfc_type: SFCType,
                               test_env: TestEnv):
        """
        Determine if the throughput requirement of the given path is meet based on current sfc type
        :param model: given model
        :param path: given path
        :param throughput: given throughput requirement
        :param cur_sfc_type: current sfc type
        :param test_env: test environment
        :return: true or false
        """
        if cur_sfc_type == SFCType.Active:
            for i in range(len(path) - 1):
                if model.topo[path[i]][path[i + 1]]["bandwidth"] - model.topo[path[i]][path[i + 1]]["reserved"] - \
                        model.topo[path[i]][path[i + 1]]["active"] < throughput:
                    return False
            return True
        else:
            for i in range(len(path) - 1):
                if test_env == TestEnv.Aggressive:
                    if model.topo[path[i]][path[i + 1]]["bandwidth"] < throughput:
                        return False
                if test_env == TestEnv.Normal or test_env == TestEnv.MaxReservation:
                    if model.topo[path[i]][path[i + 1]]["bandwidth"] - model.topo[path[i]][path[i + 1]][
                        "active"] < throughput:
                        return False
                if test_env == TestEnv.FullyReservation:
                    if model.topo[path[i]][path[i + 1]]["bandwidth"] - model.topo[path[i]][path[i + 1]]["reserved"] - \
                            model.topo[path[i]][path[i + 1]]["active"] < throughput:
                        return False
            return True

    def is_path_latency_met(self, model: Model, path_s2c: List, path_c2d: List, latency: float):
        """
        Determine if the latency requirement of the given path is meet
        :param model: given model
        :param path_s2c: given path from start server to current server
        :param path_c2d: given path from current server to destination server
        :param latency: given latency
        :return: true or false
        """
        path_latency = 0
        for i in range(len(path_s2c) - 1):
            path_latency += model.topo[path_s2c[i]][path_s2c[i + 1]]["latency"]  # calculate latency of path
        for i in range(len(path_c2d) - 1):
            path_latency += model.topo[path_c2d[i]][path_c2d[i + 1]]["latency"]  # calculate latency of path
        if path_latency <= latency:
            return True
        return False

    def verify_active(self, model: Model, cur_sfc_index: int, cur_server_index: int, test_env: TestEnv):
        """
        Verify if current active sfc can be put on current server based on following two principles
        1. if the remaining computing resource is still enough for this sfc
        2. if available paths still exist
        Both these two principles are met can return true, else false
        :param test_env:
        :param model: model
        :param cur_sfc_index: current sfc index
        :param cur_server_index: current server index
        :return: true or false
        """

        # principle 1
        if model.topo.nodes[cur_server_index]["computing_resource"] - model.topo.nodes[cur_server_index]["active"] - \
                model.topo.nodes[cur_server_index]["reserved"] < model.sfc_list[cur_sfc_index].computing_resource:
            return False

        # principle 2
        remain_latency = model.sfc_list[cur_sfc_index].latency - model.sfc_list[cur_sfc_index].process_latency
        for path_s2c in nx.all_shortest_paths(model.topo, model.sfc_list[cur_sfc_index].s, cur_server_index):
            if self.is_path_throughput_met(model, path_s2c, model.sfc_list[cur_sfc_index].tp, SFCType.Active, test_env):
                for path_c2d in nx.all_shortest_paths(model.topo, cur_server_index, model.sfc_list[cur_sfc_index].d):
                    if self.is_path_latency_met(model, path_s2c, path_c2d,
                                                remain_latency) and self.is_path_throughput_met(
                        model, path_c2d, model.sfc_list[cur_sfc_index].tp, SFCType.Active, test_env):
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

    @abstractmethod
    def generate_decision(self, model: Model, cur_sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, don't check if it can be deployed
        :param model: model
        :param cur_sfc_index: current sfc index
        :param state: state
        :param test_env: test environment
        :return: decision
        """
        return Decision()

    def make_decision(self, model: Model, cur_sfc_index: int, state: List, test_env: TestEnv):
        """
        make deploy decisions, and check up if this decision can be placed, consider no backup and with backup
        :param model: the model
        :param cur_sfc_index: cur index of sfc
        :param state: state
        :param test_env: test environment
        :return: success or failed, the real decision
        """
        decision = self.generate_decision(model, cur_sfc_index, state, test_env)
        assert decision.active_server != VariableState.Uninitialized

        # servers met or not
        if model.topo.nodes[decision.active_server]["computing_resource"] - model.topo.nodes[decision.active_server][
            "active"] - \
                model.topo.nodes[decision.active_server]["reserved"] < model.sfc_list[cur_sfc_index].computing_resource:
            return False, decision
        if test_env == TestEnv.Aggressive:
            if model.topo.nodes[decision.standby_server]["computing_resource"] < model.sfc_list[
                cur_sfc_index].computing_resource:
                return False, decision
        if test_env == TestEnv.Normal or test_env == TestEnv.MaxReservation:
            if model.topo.nodes[decision.standby_server]["computing_resource"] - \
                    model.topo.nodes[decision.standby_server][
                        "active"] < \
                    model.sfc_list[cur_sfc_index].computing_resource:
                return False, decision
        if test_env == TestEnv.FullyReservation:
            cs = model.topo.nodes[decision.standby_server]["computing_resource"]
            active = model.topo.nodes[decision.standby_server]["active"]
            reserved = model.topo.nodes[decision.standby_server]["reserved"]
            if cs - active - reserved < model.sfc_list[cur_sfc_index].computing_resource:
                return False, decision

        # paths met or not
        paths = self.select_paths(model, cur_sfc_index, decision.active_server, decision.standby_server, test_env)
        flag = paths[0]
        paths = paths[1:]
        if not flag:
            return False, decision

        def final_check():
            """
            This function check if the active server and the stand-by server is the same server, and check if
            each link is not satified because we don't jointly consider the active_s2c, active_c2d, update_path,
            standby_s2c and standby_c2d
            :return: true or false
            """
            return True
            active = dict()
            standby = dict()
            active_s2c = paths[0][0]
            active_c2d = paths[0][1]
            for i in range(1, len(active_s2c)):
                if (active_s2c[i - 1], active_s2c[i]) not in active.keys():
                    active[active_s2c[i - 1], active_s2c[i]] = 0
                active[active_s2c[i - 1], active_s2c[i]] += model.sfc_list[cur_sfc_index].tp
            for i in range(1, len(active_c2d)):
                if (active_c2d[i - 1], active_c2d[i]) not in active.keys():
                    active[active_c2d[i - 1], active_c2d[i]] = 0
                active[active_c2d[i - 1], active_c2d[i]] += model.sfc_list[cur_sfc_index].tp

            if test_env != TestEnv.NoBackup:
                update = paths[2]
                standby_s2c = paths[1][0]
                standby_c2d = paths[1][1]
                for i in range(1, len(update)):
                    if (update[i - 1], update[i]) not in active.keys():
                        active[update[i - 1], update[i]] = 0
                    active[update[i - 1], update[i]] += model.sfc_list[cur_sfc_index].update_tp

                for item in active:
                    if model.topo.edges[item]["active"] + active[item] > model.topo.edges[item]["bandwidth"]:
                        return False

                if test_env == TestEnv.FullyReservation:
                    for i in range(1, len(standby_s2c)):
                        if (standby_s2c[i - 1], standby_s2c[i]) not in standby.keys():
                            standby[standby_s2c[i - 1], standby_s2c[i]] = 0
                        standby[standby_s2c[i - 1], standby_s2c[i]] += model.sfc_list[cur_sfc_index].tp
                    for i in range(1, len(standby_c2d)):
                        if (standby_c2d[i - 1], standby_c2d[i]) not in standby.keys():
                            standby[standby_c2d[i - 1], standby_c2d[i]] = 0
                        standby[standby_c2d[i - 1], standby_c2d[i]] += model.sfc_list[cur_sfc_index].tp
                    total = dict()
                    for item in active:
                        if item not in total.keys():
                            total[item] = 0
                        total[item] += active[item]
                    for item in standby:
                        if item not in total.keys():
                            total[item] = 0
                        total[item] += standby[item]
                    for item in total:
                        if model.topo.edges[item]["active"] + total[item] + model.topo.edges[item]["reserved"] > model.topo.edges[item[0], item[1]]["bandwidth"]:
                            return False
                    return True

                if test_env == TestEnv.MaxReservation:
                    for i in range(1, len(standby_s2c)):
                        if (standby_s2c[i - 1], standby_s2c[i]) not in standby.keys():
                            standby[standby_s2c[i - 1], standby_s2c[i]] = 0
                        standby[standby_s2c[i - 1], standby_s2c[i]] = max(model.sfc_list[cur_sfc_index].tp, standby[standby_s2c[i - 1], standby_s2c[i]])
                    for i in range(1, len(standby_c2d)):
                        if (standby_c2d[i - 1], standby_c2d[i]) not in standby.keys():
                            standby[standby_c2d[i - 1], standby_c2d[i]] = 0
                        standby[standby_c2d[i - 1], standby_c2d[i]] = max(model.sfc_list[cur_sfc_index].tp, standby[standby_c2d[i - 1], standby_c2d[i]])
                    for item in active:
                        if item not in standby.keys():
                            standby[item] = 0
                        if model.topo.edges[item]["active"] + active[item] + max(model.topo.edges[item]["reserved"], standby[item]) > model.topo.edges[item[0], item[1]]["bandwidth"]:
                            return False
                    return True
                return True
            else:
                for item in active:
                    if model.topo.edges[item[0], item[1]]["active"] + active[item] > model.topo.edges[item[0], item[1]]["bandwidth"]:
                        return False
                return True

        if test_env != TestEnv.NoBackup:
            # if not final_check():
            #     flag = False
            decision.set_active_path_s2c(paths[0][0])
            decision.set_active_path_c2d(paths[0][1])
            decision.set_standby_path_s2c(paths[1][0])
            decision.set_standby_path_c2d(paths[1][1])
            decision.set_update_path(paths[2])
        else:
            # if not final_check():
            #     flag = False
            decision.set_active_path_s2c(paths[0][0])
            decision.set_active_path_c2d(paths[0][1])
        return flag, decision

    def select_path(self, path_set: List, coupled: bool):
        """
        select path from paths
        :param path_set:
        :param coupled:
        :return: if success, return the path selected, else return False
        """
        if len(path_set) == 0:
            return False

        if not coupled:
            min_value = float("inf")
            min_path = []
            for path in path_set:
                length = len(path)
                if length < min_value:
                    min_value = length
                    min_path = path
            return min_path
        else:
            min_value = float("inf")
            min_path = []
            for path_item in path_set:
                length = len(path_item[0]) + len(path_item[1])
                if length < min_value:
                    min_value = length
                    min_path = path_item
            return min_path

    def select_paths(self, model: Model, sfc_index: int, active_index: int, standby_index: int, test_env: TestEnv):
        """
        select paths for determined active instance server index and stand-by instance server index
        :param model: model
        :param sfc_index: sfc index
        :param active_index: active server index
        :param standby_index: stand-by server index
        :param test_env: test environment
        :return: true or false, select path
        """

        # No backup condition
        if test_env == TestEnv.NoBackup:
            flag = True
            active_paths = []
            temp_active_s2c = next(nx.all_shortest_paths(model.topo, model.sfc_list[sfc_index].s, active_index))
            temp_active_c2d = next(nx.all_shortest_paths(model.topo, active_index, model.sfc_list[sfc_index].d))
            for active_s2c in nx.all_shortest_paths(model.topo, model.sfc_list[sfc_index].s, active_index):
                if self.is_path_throughput_met(model, active_s2c, model.sfc_list[sfc_index].tp, SFCType.Active,
                                               test_env):
                    for active_c2d in nx.all_shortest_paths(model.topo, active_index, model.sfc_list[sfc_index].d):
                        if self.is_path_latency_met(model, active_s2c, active_c2d,
                                                    model.sfc_list[sfc_index].latency - model.sfc_list[
                                                        sfc_index].process_latency) and self.is_path_throughput_met(
                            model,
                            active_c2d,
                            model.sfc_list[
                                sfc_index].tp,
                            SFCType.Active, test_env):
                            active_paths.append([active_s2c, active_c2d])

            if len(active_paths) == 0:
                active_paths.append([temp_active_s2c, temp_active_c2d])
                flag = False
            active_path = self.select_path(active_paths, True)
            return [flag, active_path]

        # calculate paths for active instance
        flag = True
        active_paths = []
        temp_active_s2c = next(nx.all_shortest_paths(model.topo, model.sfc_list[sfc_index].s, active_index))
        temp_active_c2d = next(nx.all_shortest_paths(model.topo, active_index, model.sfc_list[sfc_index].d))
        for active_s2c in nx.all_shortest_paths(model.topo, model.sfc_list[sfc_index].s, active_index):
            if self.is_path_throughput_met(model, active_s2c, model.sfc_list[sfc_index].tp, SFCType.Active, test_env):
                for active_c2d in nx.all_shortest_paths(model.topo, active_index, model.sfc_list[sfc_index].d):
                    if self.is_path_latency_met(model, active_s2c, active_c2d,
                                                model.sfc_list[sfc_index].latency - model.sfc_list[
                                                    sfc_index].process_latency) and self.is_path_throughput_met(
                        model,
                        active_c2d,
                        model.sfc_list[sfc_index].tp, SFCType.Active, test_env):
                        active_paths.append([active_s2c, active_c2d])
        if len(active_paths) == 0:
            active_paths.append([temp_active_s2c, temp_active_c2d])
            flag = False
        active_path = self.select_path(active_paths, True)

        # calculate paths for stand-by instance
        standby_paths = []
        temp_standby_s2c = next(nx.all_shortest_paths(model.topo, model.sfc_list[sfc_index].s, standby_index))
        temp_standby_c2d = next(nx.all_shortest_paths(model.topo, standby_index, model.sfc_list[sfc_index].d))

        for standby_s2c in nx.all_shortest_paths(model.topo, model.sfc_list[sfc_index].s, standby_index):
            if self.is_path_throughput_met(model, standby_s2c, model.sfc_list[sfc_index].tp, SFCType.Standby, test_env):
                for standby_c2d in nx.all_shortest_paths(model.topo, standby_index, model.sfc_list[sfc_index].d):
                    if self.is_path_latency_met(model, standby_s2c, standby_c2d,
                                                model.sfc_list[sfc_index].latency - model.sfc_list[
                                                    sfc_index].process_latency) and self.is_path_throughput_met(
                        model,
                        standby_c2d,
                        model.sfc_list[
                            sfc_index].tp,
                        SFCType.Standby, test_env):
                        standby_paths.append([standby_s2c, standby_c2d])
        if len(standby_paths) == 0:
            flag = False
            standby_paths.append([temp_standby_s2c, temp_standby_c2d])
        standby_path = self.select_path(standby_paths, True)

        # calculate paths for updating
        update_paths = []
        temp_update = next(nx.all_shortest_paths(model.topo, active_index, standby_index))
        for path in nx.all_shortest_paths(model.topo, active_index, standby_index):
            if self.is_path_throughput_met(model, path, model.sfc_list[sfc_index].update_tp, SFCType.Active, test_env):
                update_paths.append(path)
        if len(update_paths) == 0:
            flag = False
            update_paths.append(temp_update)
        update_path = self.select_path(update_paths, False)

        return [flag, active_path, standby_path, update_path]


class RandomDecisionMakerWithGuarantee(DecisionMaker):
    """
    The class used to make random decision
    """

    def __init__(self):
        super(RandomDecisionMakerWithGuarantee, self).__init__()

    def verify_standby(self, model: Model, cur_sfc_index: int, active_server_index: int, cur_server_index: int,
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

    def narrow_decision_set(self, model: Model, cur_sfc_index: int, test_env: TestEnv):
        """
        Used to narrow available decision set
        :param test_env:
        :param model: model
        :param cur_sfc_index: cur processing sfc index
        :return: decision sets
        """
        desision_set = []
        for i in range(len(model.topo.nodes)):
            if not self.verify_active(model, cur_sfc_index, i, test_env):
                continue
            if test_env == TestEnv.NoBackup:
                desision_set.append(Decision(i, -1))
                continue
            for j in range(len(model.topo.nodes)):
                if self.verify_standby(model, cur_sfc_index, i, j, test_env) and i != j:
                    desision_set.append(Decision(i, j))
        return desision_set

    def select_decision_from_decisions(self, decisions: List):
        decision = random.sample(decisions, 1)[0]
        return decision

    def generate_decision(self, model: Model, cur_sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, don't check if it can be deployed
        :param model: model
        :param cur_sfc_index: current sfc index
        :param test_env: test environment
        :return: decision
        """
        decisions = self.narrow_decision_set(model, cur_sfc_index, test_env)
        if len(decisions) == 0:
            decision = Decision()
            decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
            decision.standby_server = random.sample(range(len(model.topo.nodes)), 1)[0]
            return decision
        decision = self.select_decision_from_decisions(decisions)
        return decision


class ICCDecisionMakerWithStrongGuarantee(DecisionMaker):
    """
    The class used to make random decision
    """

    def __init__(self):
        super(ICCDecisionMakerWithStrongGuarantee, self).__init__()

    def select_decision_from_decisions(self, decisions: List):
        decision = random.sample(decisions, 1)[0]
        return decision

    def select_paths_for_updating(self, model: Model, sfc_index: int, active_index: int, standby_index: int, test_env: TestEnv):
        """
        select paths for determined active instance server index and stand-by instance server index
        :param model: model
        :param sfc_index: sfc index
        :param active_index: active server index
        :param standby_index: stand-by server index
        :param test_env: test environment
        :return: true or false, select path
        """

        # No backup condition
        assert test_env != TestEnv.NoBackup

        # calculate paths for updating
        update_paths = []
        temp_update = next(nx.all_shortest_paths(model.topo, active_index, standby_index))
        for path in nx.all_shortest_paths(model.topo, active_index, standby_index):
            if self.is_path_throughput_met(model, path, model.sfc_list[sfc_index].update_tp, SFCType.Active, test_env):
                update_paths.append(path)
        if len(update_paths) == 0:
            update_paths.append(temp_update)
        update_path = self.select_path(update_paths, False)
        return update_path

    def judge_decision(self, model: Model, cur_active_index: int, cur_standby_index: int, cur_sfc_index: int, test_env: TestEnv):
        """
        Used to narrow available decision set
        :param test_env:
        :param model: model
        :param cur_sfc_index: cur processing sfc index
        :return: decision sets
        """
        if cur_active_index != cur_standby_index and self.verify_active(model, cur_sfc_index, cur_active_index, test_env) and self.verify_standby(model, cur_sfc_index, cur_active_index, cur_standby_index, test_env):
            return True
        return False

    def generate_decision(self, model: Model, cur_sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, don't check if it can be deployed
        :param model: model
        :param cur_sfc_index: current sfc index
        :param test_env: test environment
        :return: decision
        """
        decisions = []
        for i in range(len(model.topo.nodes)):
            for j in range(len(model.topo.nodes)):
                decision = Decision()
                decision.active_server = i
                decision.standby_server = j
                decision.flag = self.judge_decision(model, i, j, cur_sfc_index, test_env)
                decision.update_path = self.select_paths_for_updating(model, cur_sfc_index, i, j, test_env)
                decisions.append(decision)

        decisions = sorted(decisions, key=lambda x: (model.topo.nodes[x.active_server]['computing_resource'] -
                                                         model.topo.nodes[x.active_server]['active'], len(x.update_path)))

        for decision in decisions:
            if decision.flag:
                return decision
        decision = Decision()
        decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
        decision.standby_server = random.sample(range(len(model.topo.nodes)), 1)[0]
        return decision


class WorstDecisionMakerWithStrongGuarantee(DecisionMaker):
    """
    The class used to make random decision
    """

    def __init__(self):
        super(WorstDecisionMakerWithStrongGuarantee, self).__init__()

    def select_decision_from_decisions(self, decisions: List):
        decision = random.sample(decisions, 1)[0]
        return decision

    def select_paths_for_updating(self, model: Model, sfc_index: int, active_index: int, standby_index: int, test_env: TestEnv):
        """
        select paths for determined active instance server index and stand-by instance server index
        :param model: model
        :param sfc_index: sfc index
        :param active_index: active server index
        :param standby_index: stand-by server index
        :param test_env: test environment
        :return: true or false, select path
        """

        # No backup condition
        assert test_env != TestEnv.NoBackup

        # calculate paths for updating
        update_paths = []
        temp_update = next(nx.all_shortest_paths(model.topo, active_index, standby_index))
        for path in nx.all_shortest_paths(model.topo, active_index, standby_index):
            if self.is_path_throughput_met(model, path, model.sfc_list[sfc_index].update_tp, SFCType.Active, test_env):
                update_paths.append(path)
        if len(update_paths) == 0:
            update_paths.append(temp_update)
        update_path = self.select_path(update_paths, False)
        return update_path

    def judge_decision(self, model: Model, cur_active_index: int, cur_standby_index: int, cur_sfc_index: int, test_env: TestEnv):
        """
        Used to narrow available decision set
        :param test_env:
        :param model: model
        :param cur_sfc_index: cur processing sfc index
        :return: decision sets
        """
        if cur_active_index != cur_standby_index and self.verify_active(model, cur_sfc_index, cur_active_index, test_env) and self.verify_standby(model, cur_sfc_index, cur_active_index, cur_standby_index, test_env):
            return True
        return False

    def generate_decision(self, model: Model, cur_sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, don't check if it can be deployed
        :param model: model
        :param cur_sfc_index: current sfc index
        :param test_env: test environment
        :return: decision
        """
        decisions = []
        for i in range(len(model.topo.nodes)):
            for j in range(len(model.topo.nodes)):
                decision = Decision()
                decision.active_server = i
                decision.standby_server = j
                decision.flag = self.judge_decision(model, i, j, cur_sfc_index, test_env)
                decision.update_path = self.select_paths_for_updating(model, cur_sfc_index, i, j, test_env)
                decisions.append(decision)

        decisions = sorted(decisions, key=lambda x: ( - model.topo.nodes[x.active_server]['computing_resource'] +
                                                         model.topo.nodes[x.active_server]['active'], len(x.update_path)))

        for decision in decisions:
            if decision.flag:
                return decision
        decision = Decision()
        decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
        decision.standby_server = random.sample(range(len(model.topo.nodes)), 1)[0]
        return decision


class RandomDecisionMakerWithStrongGuarantee(DecisionMaker):
    """
    The class used to make random decision
    """

    def __init__(self):
        super(RandomDecisionMakerWithStrongGuarantee, self).__init__()

    def narrow_decision_set(self, model: Model, cur_sfc_index: int, test_env: TestEnv):
        """
        Used to narrow available decision set
        :param test_env:
        :param model: model
        :param cur_sfc_index: cur processing sfc index
        :return: decision sets
        """
        desision_set = []
        for i in range(len(model.topo.nodes)):
            if not self.verify_active(model, cur_sfc_index, i, test_env):
                continue
            if test_env == TestEnv.NoBackup:
                desision_set.append(Decision(i, -1))
                continue
            for j in range(len(model.topo.nodes)):
                if self.verify_standby(model, cur_sfc_index, i, j, test_env) and i != j:
                    desision_set.append(Decision(i, j))
        return desision_set

    def select_decision_from_decisions(self, decisions: List):
        decision = random.sample(decisions, 1)[0]
        return decision

    def generate_decision(self, model: Model, cur_sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, don't check if it can be deployed
        :param model: model
        :param cur_sfc_index: current sfc index
        :param test_env: test environment
        :return: decision
        """
        decisions = self.narrow_decision_set(model, cur_sfc_index, test_env)

        all_active = [0 for _ in range(len(model.topo.nodes))]
        for d in decisions:
            if all_active[d.active_server] == 0:
                all_active[d.active_server] = 1

        if len(decisions) == 0:
            decision = Decision()
            decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
            decision.standby_server = random.sample(range(len(model.topo.nodes)), 1)[0]
            return decision
        decision = self.select_decision_from_decisions(decisions)
        return decision


class RandomDecisionMaker(DecisionMaker):
    """
    The class used to make random decision
    """

    def __init__(self):
        super(RandomDecisionMaker, self).__init__()

    def generate_decision(self, model: Model, cur_sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, don't check if it can be deployed
        :param model: model
        :param cur_sfc_index: current sfc index
        :param test_env: test environment
        :return: decision
        """
        decision = Decision()
        decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
        decision.standby_server = random.sample(range(len(model.topo.nodes)), 1)[0]
        return decision


class ICCheuristic(DecisionMaker):
    """
    The class used to make greedy decision of ICC paper raised
    """

    def __init__(self):
        super(ICCheuristic, self).__init__()

    def verify_standby(self, model: Model, cur_sfc_index: int, active_server_index: int, cur_server_index: int,
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

    def select_decision_from_decisions(self, decisions: List):
        # decisions = decisions.sort()
        decision = random.sample(decisions, 1)[0]
        return decision

    def select_paths_for_updating(self, model: Model, sfc_index: int, active_index: int, standby_index: int, test_env: TestEnv):
        """
        select paths for determined active instance server index and stand-by instance server index
        :param model: model
        :param sfc_index: sfc index
        :param active_index: active server index
        :param standby_index: stand-by server index
        :param test_env: test environment
        :return: true or false, select path
        """

        # No backup condition
        assert test_env != TestEnv.NoBackup

        # calculate paths for updating
        update_paths = []
        temp_update = next(nx.all_shortest_paths(model.topo, active_index, standby_index))
        for path in nx.all_shortest_paths(model.topo, active_index, standby_index):
            if self.is_path_throughput_met(model, path, model.sfc_list[sfc_index].update_tp, SFCType.Active, test_env):
                update_paths.append(path)
        if len(update_paths) == 0:
            update_paths.append(temp_update)
        update_path = self.select_path(update_paths, False)
        return update_path

    def judge_decision(self, model: Model, cur_active_index: int, cur_standby_index: int, cur_sfc_index: int, test_env: TestEnv):
        """
        Used to narrow available decision set
        :param test_env:
        :param model: model
        :param cur_sfc_index: cur processing sfc index
        :return: decision sets
        """
        if cur_active_index != cur_standby_index and self.verify_active(model, cur_sfc_index, cur_active_index, test_env) and self.verify_standby(model, cur_sfc_index, cur_active_index, cur_standby_index, test_env):
            return True
        return False

    def generate_decision(self, model: Model, cur_sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, don't check if it can be deployed
        :param model: model
        :param cur_sfc_index: current sfc index
        :param test_env: test environment
        :return: decision
        """
        decisions = []
        for i in range(len(model.topo.nodes)):
            for j in range(len(model.topo.nodes)):
                decision = Decision()
                decision.active_server = i
                decision.standby_server = j
                decision.flag = self.judge_decision(model, i, j, cur_sfc_index, test_env)
                decision.update_path = self.select_paths_for_updating(model, cur_sfc_index, i, j, test_env)
                decisions.append(decision)

        decisions = sorted(decisions, key=lambda x: (model.topo.nodes[x.active_server]['computing_resource'] -
                                                         model.topo.nodes[x.active_server]['active'], len(x.update_path)))

        for decision in decisions:
            if decision.flag:
                return decision
        decision = Decision()
        decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
        decision.standby_server = random.sample(range(len(model.topo.nodes)), 1)[0]
        return decision

class Worst(DecisionMaker):
    """
    The class used to make greedy decision of ICC paper raised
    """

    def __init__(self):
        super(Worst, self).__init__()

    def verify_standby(self, model: Model, cur_sfc_index: int, active_server_index: int, cur_server_index: int,
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

    def select_decision_from_decisions(self, decisions: List):
        # decisions = decisions.sort()
        decision = random.sample(decisions, 1)[0]
        return decision


    def select_paths_for_updating(self, model: Model, sfc_index: int, active_index: int, standby_index: int, test_env: TestEnv):
        """
        select paths for determined active instance server index and stand-by instance server index
        :param model: model
        :param sfc_index: sfc index
        :param active_index: active server index
        :param standby_index: stand-by server index
        :param test_env: test environment
        :return: true or false, select path
        """

        # No backup condition
        assert test_env != TestEnv.NoBackup

        # calculate paths for updating
        update_paths = []
        temp_update = next(nx.all_shortest_paths(model.topo, active_index, standby_index))
        for path in nx.all_shortest_paths(model.topo, active_index, standby_index):
            if self.is_path_throughput_met(model, path, model.sfc_list[sfc_index].update_tp, SFCType.Active, test_env):
                update_paths.append(path)
        if len(update_paths) == 0:
            update_paths.append(temp_update)
        update_path = self.select_path(update_paths, False)
        return update_path

    def judge_decision(self, model: Model, cur_active_index: int, cur_standby_index: int, cur_sfc_index: int, test_env: TestEnv):
        """
        Used to narrow available decision set
        :param test_env:
        :param model: model
        :param cur_sfc_index: cur processing sfc index
        :return: decision sets
        """
        if cur_active_index != cur_standby_index and self.verify_active(model, cur_sfc_index, cur_active_index, test_env) and self.verify_standby(model, cur_sfc_index, cur_active_index, cur_standby_index, test_env):
            return True
        return False

    def generate_decision(self, model: Model, cur_sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, don't check if it can be deployed
        :param model: model
        :param cur_sfc_index: current sfc index
        :param test_env: test environment
        :return: decision
        """
        decisions = []
        for i in range(len(model.topo.nodes)):
            for j in range(len(model.topo.nodes)):
                decision = Decision()
                decision.active_server = i
                decision.standby_server = j
                decision.flag = self.judge_decision(model, i, j, cur_sfc_index, test_env)
                decision.update_path = self.select_paths_for_updating(model, cur_sfc_index, i, j, test_env)
                decisions.append(decision)

        decisions = sorted(decisions, key=lambda x: ( - model.topo.nodes[x.active_server]['computing_resource'] +
                                                         model.topo.nodes[x.active_server]['active'], len(x.update_path)))

        for decision in decisions:
            if decision.flag:
                return decision
        decision = Decision()
        decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
        decision.standby_server = random.sample(range(len(model.topo.nodes)), 1)[0]
        return decision


# test
def main():
    # import generator
    # topo = generator.generate_topology(30)
    # for path in nx.all_shortest_paths(topo, 15, 16):
    #     print(path)
    # nx.draw(topo, with_labels=True)
    # plt.show()
    print(random.sample([1], 1))


if __name__ == '__main__':
    main()
