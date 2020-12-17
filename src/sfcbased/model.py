from typing import List, Callable
import networkx as nx
from enum import Enum, unique
import random
from abc import abstractmethod
import math


class VirtualException(BaseException):
    def __init__(self, _type, _func):
        BaseException(self)


class BaseObject(object):
    def __repr__(self):
        """
        when function print() is called, this function will determine what to display

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
    def __init__(self, active_server: int = VariableState.Uninitialized,
                 standby_server: int = VariableState.Uninitialized):
        self.flag = VariableState.Uninitialized  # flag determines if this decision satisfy the requirement of specific algorithm
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
    designed for Monitoring the actions of whole system
    """

    action_list = []
    format_logs = []

    @classmethod
    def state_transition(cls, time: int, sfc_index: int, pre_state: State, new_state: State,
                         reason: BrokenReason = BrokenReason.NoReason):
        """
        handle the state transition of sfc

        why use BrokenReason? because there are many conditions when the state of sfc transits to Broken,
        we should identify them.

        :param time: occur time
        :param sfc_index: sfc index
        :param pre_state: previous state
        :param new_state: new state
        :param reason: the broken reason
        :return: None
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
    this class mainly used to determine whether the failed instances are active or standby
    when generate them
    """

    def __init__(self, sfc_index: int, is_active: bool):
        self.sfc_index = sfc_index
        self.is_active = is_active

    def __str__(self):
        if self.is_active:
            return "active instance of SFC {}".format(self.sfc_index)
        else:
            return "standby instance of SFC {}".format(self.sfc_index)


class ACSFC(BaseObject):
    """
    this class is denoted as an active SFC.
    """

    def __init__(self):
        self.server = VariableState.Uninitialized
        self.starttime = VariableState.Uninitialized
        self.downtime = VariableState.Uninitialized
        self.path_s2c = VariableState.Uninitialized
        self.path_c2d = VariableState.Uninitialized


class SBSFC(BaseObject):
    """
    this class is denoted as a standby SFC.
    """

    def __init__(self):
        self.server = VariableState.Uninitialized
        self.starttime = VariableState.Uninitialized
        self.downtime = VariableState.Uninitialized
        self.path_s2c = VariableState.Uninitialized
        self.path_c2d = VariableState.Uninitialized


class SFC(BaseObject):
    """
    this class is denoted as a SFC
    """

    def __init__(self, computing_resource: float, tp: float, latency: float, update_tp: float, process_latency: float,
                 s: int,
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
        self.arriving_time = time
        self.TTL = TTL
        self.failed_num = 0

        self.state = State.Undeployed
        self.update_path = VariableState.Uninitialized
        self.active_sfc = ACSFC()
        self.standby_sfc = SBSFC()

    @property
    def time(self):
        """
        this property represents the real deploy time

        :return: float
        """
        return self.arriving_time + math.pow(2, self.failed_num) - 1

    def __str__(self):
        """
        display in console with specified format.

        :return: str
        """
        return "(computing_resource: {}, throughput: {}, latency: {}, update throughput: {}, process latency: {}, from {}->{}, time: {}, TTL: {})".format(
            self.computing_resource,
            self.tp,
            self.latency, self.update_tp,
            self.process_latency,
            self.s, self.d, self.arriving_time, self.TTL)

    def set_state(self, cur_time: int, sfc_index: int, new_state: State, reason: BrokenReason = BrokenReason.NoReason):
        """
        setting up new state

        :param sfc_index:
        :param reason:
        :param cur_time: occur time
        :param new_state: new state
        :return: None
        """
        # Monitor the state transition
        Monitor.state_transition(cur_time, sfc_index, self.state, new_state, reason)

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
            self.active_sfc.downtime = cur_time
            if new_state == State.Backup:
                self.standby_sfc.starttime = cur_time
            if new_state == State.Broken:
                self.standby_sfc.starttime = -1
                self.standby_sfc.downtime = -1
        if self.state == State.Backup:
            if new_state == State.Broken:
                if reason == BrokenReason.TimeExpired:
                    self.standby_sfc.downtime = self.time + self.TTL
                else:
                    self.standby_sfc.downtime = cur_time
        if self.state == State.Failed:
            if new_state == State.Normal:
                self.active_sfc.starttime = self.time

        self.state = new_state


class Model(BaseObject):
    """
    this class is denoted as the model

    a model contains following:
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
        print out the start time and down time of each instance of each sfc
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
        calculate throughput

        :return: float throughput
        """
        return sum(sfc.tp for sfc in self.sfc_list if sfc.state != State.Failed)

    def calculate_total_reward(self):
        """
        calculate total reward

        :return: int total reward
        """
        total_success = 0
        total_failed = 0
        for sfc in self.sfc_list:
            total_failed += sfc.failed_num
            if sfc.state != State.Failed:
                total_success += 1

        return total_success - total_failed

    def calculate_place_num(self):
        """
        calculate total place num

        :return: int placement number
        """
        total_success = 0
        total_failed = 0
        for sfc in self.sfc_list:
            total_failed += sfc.failed_num
            if sfc.state != State.Failed:
                total_success += 1

        return total_success + total_failed

    def calculate_place_cdf(self, num=10):
        """
        calculate place cdf

        :param num: max plot place number
        :return: List[float] cdf
        """
        res = [0 for _ in range(num + 1)]

        for sfc in self.sfc_list:
            if sfc.failed_num < num:
                if sfc.state == State.Failed:
                    res[sfc.failed_num] += 1
                else:
                    res[sfc.failed_num + 1] += 1

        total = sum(res)
        for i in range(1, num + 1):
            res[i] += res[i - 1]

        for i in range(num + 1):
            res[i] /= total

        return res

    def calculate_accepted_number(self):
        """
        calculate accepted number

        :return: int accepted number
        """
        return sum(1 for sfc in self.sfc_list if sfc.state != State.Failed)

    def calculate_service_availability(self):
        """
        calculate service availability

        :return: float service availability
        """
        return sum(
            sfc.active_sfc.downtime - sfc.active_sfc.starttime + sfc.standby_sfc.downtime - sfc.standby_sfc.starttime
            for sfc in self.sfc_list if sfc.state == State.Broken)

    def calculate_fail_rate(self):
        """
        calculate fail rate

        :return: float fail rate
        """
        real_service = 0
        should_service = 0
        for i in range(len(self.sfc_list)):
            cur_sfc = self.sfc_list[i]
            if cur_sfc.state == State.Broken:
                should_service += cur_sfc.time + cur_sfc.TTL - cur_sfc.active_sfc.downtime
                real_service += cur_sfc.standby_sfc.downtime - cur_sfc.standby_sfc.starttime
        if should_service == 0:
            return 0
        return 1 - real_service / should_service

    def calculate_accept_rate(self):
        """
        calculate accept rate

        :return: float accept rate
        """
        return 1 - sum(1 for item in self.sfc_list if item.state == State.Failed) / len(self.sfc_list)

    def calculate_server_occupied_rate(self):
        """
        calculate server occupied rate

        :return: [dict] server occupied rate
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
        calculate link occupied rate

        :return: [dict] link occupied rate
        """
        link_rate = []
        for edge in self.topo.edges(data=True):
            link = dict()
            link["active"] = edge[2]['active'] / edge[2]['bandwidth']
            link["reserved"] = edge[2]['reserved'] / edge[2]['bandwidth']
            link_rate.append(link)
        return link_rate


path_database = dict()


def all_shortest_paths(topo: nx.Graph, src: int, dst: int):
    global path_database
    if (src, dst) in path_database:
        return path_database[(src, dst)]
    else:
        paths = list(nx.all_shortest_paths(topo, src, dst))
        path_database[(src, dst)] = paths
        return paths


def is_path_throughput_met(model: Model, path: List, throughput: float, cur_sfc_type: SFCType,
                           test_env: TestEnv):
    """
    determine if the throughput requirement of the given path is meet based on current sfc type

    :param model: given model
    :param path: given path
    :param throughput: given throughput requirement
    :param cur_sfc_type: current sfc type
    :param test_env: test environment
    :return: bool
    """
    if cur_sfc_type == SFCType.Active:
        for i in range(len(path) - 1):
            if model.topo[path[i]][path[i + 1]]["bandwidth"] - model.topo[path[i]][path[i + 1]]["reserved"] - \
                    model.topo[path[i]][path[i + 1]]["active"] < throughput:
                return False
        return True
    else:
        assert test_env != TestEnv.NoBackup
        for i in range(len(path) - 1):
            if test_env == TestEnv.Aggressive:
                if model.topo[path[i]][path[i + 1]]["bandwidth"] < throughput:
                    return False
            elif test_env == TestEnv.Normal or test_env == TestEnv.MaxReservation:
                if model.topo[path[i]][path[i + 1]]["bandwidth"] - model.topo[path[i]][path[i + 1]][
                    "active"] < throughput:
                    return False
            else:
                if model.topo[path[i]][path[i + 1]]["bandwidth"] - model.topo[path[i]][path[i + 1]]["reserved"] - \
                        model.topo[path[i]][path[i + 1]]["active"] < throughput:
                    return False
        return True


def is_path_latency_met(model: Model, path_s2c: List, path_c2d: List, latency: float):
    """
    determine if the flow latency requirement of the given path pair is met

    :param model: given model
    :param path_s2c: given path from start server to current server
    :param path_c2d: given path from current server to destination server
    :param latency: given latency
    :return: bool
    """
    path_latency = 0
    for i in range(len(path_s2c) - 1):
        path_latency += model.topo[path_s2c[i]][path_s2c[i + 1]]["latency"]  # calculate latency of path
    for i in range(len(path_c2d) - 1):
        path_latency += model.topo[path_c2d[i]][path_c2d[i + 1]]["latency"]  # calculate latency of path
    if path_latency <= latency:
        return True
    return False


def verify_active(model: Model, sfc_index: int, cur_server_index: int, test_env: TestEnv):
    """
    verify if current active sfc can be put on current server

    this is based on following two principles
    1. if the remaining computing resource is still enough for this sfc
    2. if available paths still exist
    both these two principles are met can return true, else false

    :param model: model
    :param sfc_index: current sfc index
    :param cur_server_index: current server index
    :param test_env: test environment
    :return: bool
    """

    # principle 1
    server_computing_resource = model.topo.nodes[cur_server_index]["computing_resource"]
    server_active = model.topo.nodes[cur_server_index]["active"]
    server_reserved = model.topo.nodes[cur_server_index]["reserved"]
    computing_resource_demand = model.sfc_list[sfc_index].computing_resource
    if server_computing_resource - server_active - server_reserved < computing_resource_demand:
        return False

    # principle 2
    remain_latency = model.sfc_list[sfc_index].latency - model.sfc_list[sfc_index].process_latency
    for path_s2c in all_shortest_paths(model.topo, model.sfc_list[sfc_index].s, cur_server_index):
        if is_path_throughput_met(model, path_s2c, model.sfc_list[sfc_index].tp, SFCType.Active, test_env):
            for path_c2d in all_shortest_paths(model.topo, cur_server_index, model.sfc_list[sfc_index].d):
                if is_path_latency_met(model, path_s2c, path_c2d,
                                       remain_latency) and is_path_throughput_met(
                    model, path_c2d, model.sfc_list[sfc_index].tp, SFCType.Active, test_env):
                    return True
    return False


def verify_standby_computing_resource_principle(model: Model, sfc_index: int, cur_server_index: int,
                                                test_env: TestEnv):
    """
    verify standby instance for computing resource

    :param model: model
    :param sfc_index: current sfc index
    :param cur_server_index: current server index
    :param test_env: test environment
    :return: bool
    """
    assert test_env != TestEnv.NoBackup

    computing_resource = model.topo.nodes[cur_server_index]["computing_resource"]
    computing_resource_demand = model.sfc_list[sfc_index].computing_resource
    active = model.topo.nodes[cur_server_index]["active"]
    reserved = model.topo.nodes[cur_server_index]["reserved"]

    if test_env == TestEnv.Aggressive:
        if computing_resource < computing_resource_demand:
            return False
    elif test_env == TestEnv.Normal or test_env == TestEnv.MaxReservation:
        if computing_resource - active < computing_resource_demand:
            return False
    else:
        if computing_resource - active - reserved < computing_resource_demand:
            return False
    return True


def verify_standby_updating_principle(model: Model, sfc_index: int, active_index: int, cur_server_index: int,
                                      test_env: TestEnv):
    """
    verify standby instance for updating

    :param model: model
    :param sfc_index: current sfc index
    :param active_index: active server index
    :param cur_server_index: current server index
    :param test_env: test environment
    :return: bool
    """
    assert test_env != TestEnv.NoBackup

    for path in all_shortest_paths(model.topo, active_index, cur_server_index):
        if is_path_throughput_met(model, path, model.sfc_list[sfc_index].update_tp, SFCType.Active,
                                  test_env):
            return True
    return False


def verify_standby_flow_routing_principle(model: Model, sfc_index: int, cur_server_index: int, test_env: TestEnv):
    """
    verify standby instance for flow routing

    :param model: model
    :param sfc_index: current sfc index
    :param cur_server_index: current server index
    :param test_env: test environment
    :return: bool
    """
    assert test_env != TestEnv.NoBackup

    for path_s2c in all_shortest_paths(model.topo, model.sfc_list[sfc_index].s, cur_server_index):
        if is_path_throughput_met(model, path_s2c, model.sfc_list[sfc_index].tp, SFCType.Standby,
                                  test_env):
            for path_c2d in all_shortest_paths(model.topo, cur_server_index, model.sfc_list[sfc_index].d):
                if is_path_latency_met(model, path_s2c, path_c2d,
                                       model.sfc_list[sfc_index].latency - model.sfc_list[
                                           sfc_index].process_latency) and is_path_throughput_met(
                    model,
                    path_c2d,
                    model.sfc_list[
                        sfc_index].tp,
                    SFCType.Standby, test_env):
                    return True
    return False


def verify_standby(model: Model, sfc_index: int, active_index: int, cur_server_index: int, test_env: TestEnv):
    """
    verify if current standby sfc can be put on current server

    mainly based on following three principles:
    1. computing_resource: if the remaining computing resource is still enough for this sfc
    2. update: if available paths for updating still exist
    3. flow: if available paths still exist
    all these three principles are met can return true, else false
    when the active instance is deployed, the topology will change and some constraints may not be met,
    but this is just a really small case so that we don't have to consider it

    :param model: model
    :param sfc_index: current sfc index
    :param active_index: active server index
    :param cur_server_index: current server index
    :param test_env: test environment
    :return: bool
    """
    assert test_env != TestEnv.NoBackup

    if verify_standby_computing_resource_principle(model, sfc_index, cur_server_index, test_env) and \
            verify_standby_updating_principle(model, sfc_index, active_index, cur_server_index, test_env) and \
            verify_standby_flow_routing_principle(model, sfc_index, cur_server_index, test_env):
        return True
    return False


def verify_standby_relax_flow_routing(model: Model, sfc_index: int, active_index: int, cur_server_index: int,
                                      test_env: TestEnv):
    """
    verify if current standby sfc can be put on current server

    mainly based on following two principles:
    1. computing_resource: if the remaining computing resource is still enough for this sfc
    2. update: if available paths for updating still exist
    both of two principles are met can return true, else false
    when the active instance is deployed, the topology will change and some constraints may not be met,
    but this is just a really small case so that we don't have to consider it

    :param model: model
    :param sfc_index: current sfc index
    :param active_index: active server index
    :param cur_server_index: current server index
    :param test_env: test environment
    :return: bool
    """
    assert test_env != TestEnv.NoBackup

    if verify_standby_computing_resource_principle(model, sfc_index, cur_server_index, test_env) and \
            verify_standby_updating_principle(model, sfc_index, active_index, cur_server_index, test_env):
        return True
    return False


def select_paths(model: Model, sfc_index: int, active_index: int, standby_index: int, test_env: TestEnv):
    """
    select paths for determined active server index and standby server index

    :param model: model
    :param sfc_index: sfc index
    :param active_index: active server index
    :param standby_index: standby server index
    :param test_env: test environment
    :return: [bool, [[], []] ] or [bool, [[], []], [[], []], []]
    """

    # calculate paths for active instance
    flag = True
    temp_active_s2c = all_shortest_paths(model.topo, model.sfc_list[sfc_index].s, active_index)[0]
    temp_active_c2d = all_shortest_paths(model.topo, active_index, model.sfc_list[sfc_index].d)[0]

    active_paths = []
    for active_s2c in all_shortest_paths(model.topo, model.sfc_list[sfc_index].s, active_index):
        if is_path_throughput_met(model, active_s2c, model.sfc_list[sfc_index].tp, SFCType.Active,
                                  test_env):
            for active_c2d in all_shortest_paths(model.topo, active_index, model.sfc_list[sfc_index].d):
                if is_path_latency_met(model, active_s2c, active_c2d,
                                       model.sfc_list[sfc_index].latency - model.sfc_list[
                                           sfc_index].process_latency) and is_path_throughput_met(
                    model,
                    active_c2d,
                    model.sfc_list[
                        sfc_index].tp,
                    SFCType.Active, test_env):
                    active_paths.append([active_s2c, active_c2d])
    if len(active_paths) == 0:
        active_paths.append([temp_active_s2c, temp_active_c2d])
        flag = False
    active_path = random.sample(active_paths, 1)[0]

    # no backup condition
    if test_env == TestEnv.NoBackup:
        return flag, active_path

    # other conditions
    temp_standby_s2c = all_shortest_paths(model.topo, model.sfc_list[sfc_index].s, standby_index)[0]
    temp_standby_c2d = all_shortest_paths(model.topo, standby_index, model.sfc_list[sfc_index].d)[0]
    temp_update = all_shortest_paths(model.topo, active_index, standby_index)[0]

    # calculate paths for stand-by instance
    if not flag:
        return False, (temp_active_s2c, temp_active_c2d), (temp_standby_s2c, temp_standby_c2d), temp_update

    standby_paths = []
    for standby_s2c in all_shortest_paths(model.topo, model.sfc_list[sfc_index].s, standby_index):
        if is_path_throughput_met(model, standby_s2c, model.sfc_list[sfc_index].tp, SFCType.Standby, test_env):
            for standby_c2d in all_shortest_paths(model.topo, standby_index, model.sfc_list[sfc_index].d):
                if is_path_latency_met(model, standby_s2c, standby_c2d,
                                       model.sfc_list[sfc_index].latency - model.sfc_list[
                                           sfc_index].process_latency) and is_path_throughput_met(
                    model,
                    standby_c2d,
                    model.sfc_list[
                        sfc_index].tp,
                    SFCType.Standby, test_env):
                    standby_paths.append([standby_s2c, standby_c2d])
    if len(standby_paths) == 0:
        return False, (temp_active_s2c, temp_active_c2d), (temp_standby_s2c, temp_standby_c2d), temp_update

    standby_path = random.sample(standby_paths, 1)[0]

    # calculate paths for updating
    flag, update_path = select_paths_for_updating(model, sfc_index, active_index, standby_index, test_env)

    if flag:
        return True, active_path, standby_path, update_path
    else:
        return False, (temp_active_s2c, temp_active_c2d), (temp_standby_s2c, temp_standby_c2d), temp_update


class DecisionMaker(BaseObject):
    """
    this class used to make deploy decision
    """

    def __init__(self):
        super(DecisionMaker, self).__init__()

    @abstractmethod
    def generate_decision(self, model: Model, sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, needn't to check if it can be deployed

        :param model: model
        :param sfc_index: current sfc index
        :param state: state
        :param test_env: test environment
        :return: Decision
        """
        return Decision()


def make_decision(model: Model, decision_maker: DecisionMaker, sfc_index: int, state: List, test_env: TestEnv):
    """
    make deploy decisions, and check if this decision can be placed, consider no backup and with backup

    the difference between the verify part of this function and global verify function is:
    1. this part only verify the path of given decision is valid
    2. global part considers all the available path is valid

    :param model: the model
    :param decision_maker: the decision maker
    :param sfc_index: cur index of sfc
    :param state: state
    :param test_env: test environment
    :return: (bool, Decision)
    """
    decision = decision_maker.generate_decision(model, sfc_index, state, test_env)
    assert decision.active_server != VariableState.Uninitialized

    active = model.topo.nodes[decision.active_server]

    # servers met or not
    # check active server
    if active["computing_resource"] - active["active"] - active["reserved"] < model.sfc_list[
        sfc_index].computing_resource:
        return False, decision

    # check standby server
    if test_env != TestEnv.NoBackup:
        standby = model.topo.nodes[decision.standby_server]
        computing_resource = standby["computing_resource"]
        computing_resource_demand = model.sfc_list[sfc_index].computing_resource
        active_area = standby["active"]
        reserved_area = standby["reserved"]

        if test_env == TestEnv.Aggressive:
            if computing_resource < computing_resource_demand:
                return False, decision
        if test_env == TestEnv.Normal or test_env == TestEnv.MaxReservation:
            if computing_resource - active_area < computing_resource_demand:
                return False, decision
        else:
            if computing_resource - active_area - reserved_area < computing_resource_demand:
                return False, decision

    # paths met or not
    if test_env == TestEnv.NoBackup:
        flag, paths = select_paths(model, sfc_index, decision.active_server, decision.standby_server, test_env)
    else:
        flag, *paths = select_paths(model, sfc_index, decision.active_server, decision.standby_server, test_env)

    if not flag:
        return False, decision

    def final_check():
        """
        this function conducts final check

        mainly based on:
        1. if the active server and the standby server is the same server
        2. if each link is satisfied because we don't jointly consider the active_s2c, active_c2d, update_path,
            standby_s2c and standby_c2d

        :return: bool
        """
        return True
        active = dict()
        standby = dict()
        active_s2c = paths[0][0]
        active_c2d = paths[0][1]
        for i in range(1, len(active_s2c)):
            if (active_s2c[i - 1], active_s2c[i]) not in active.keys():
                active[active_s2c[i - 1], active_s2c[i]] = 0
            active[active_s2c[i - 1], active_s2c[i]] += model.sfc_list[sfc_index].tp
        for i in range(1, len(active_c2d)):
            if (active_c2d[i - 1], active_c2d[i]) not in active.keys():
                active[active_c2d[i - 1], active_c2d[i]] = 0
            active[active_c2d[i - 1], active_c2d[i]] += model.sfc_list[sfc_index].tp

        if test_env != TestEnv.NoBackup:
            update = paths[2]
            standby_s2c = paths[1][0]
            standby_c2d = paths[1][1]
            for i in range(1, len(update)):
                if (update[i - 1], update[i]) not in active.keys():
                    active[update[i - 1], update[i]] = 0
                active[update[i - 1], update[i]] += model.sfc_list[sfc_index].update_tp

            for item in active:
                if model.topo.edges[item]["active"] + active[item] > model.topo.edges[item]["bandwidth"]:
                    return False

            if test_env == TestEnv.FullyReservation:
                for i in range(1, len(standby_s2c)):
                    if (standby_s2c[i - 1], standby_s2c[i]) not in standby.keys():
                        standby[standby_s2c[i - 1], standby_s2c[i]] = 0
                    standby[standby_s2c[i - 1], standby_s2c[i]] += model.sfc_list[sfc_index].tp
                for i in range(1, len(standby_c2d)):
                    if (standby_c2d[i - 1], standby_c2d[i]) not in standby.keys():
                        standby[standby_c2d[i - 1], standby_c2d[i]] = 0
                    standby[standby_c2d[i - 1], standby_c2d[i]] += model.sfc_list[sfc_index].tp
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
                    if model.topo.edges[item]["active"] + total[item] + model.topo.edges[item]["reserved"] > \
                            model.topo.edges[item[0], item[1]]["bandwidth"]:
                        return False
                return True

            if test_env == TestEnv.MaxReservation:
                for i in range(1, len(standby_s2c)):
                    if (standby_s2c[i - 1], standby_s2c[i]) not in standby.keys():
                        standby[standby_s2c[i - 1], standby_s2c[i]] = 0
                    standby[standby_s2c[i - 1], standby_s2c[i]] = max(model.sfc_list[sfc_index].tp,
                                                                      standby[standby_s2c[i - 1], standby_s2c[i]])
                for i in range(1, len(standby_c2d)):
                    if (standby_c2d[i - 1], standby_c2d[i]) not in standby.keys():
                        standby[standby_c2d[i - 1], standby_c2d[i]] = 0
                    standby[standby_c2d[i - 1], standby_c2d[i]] = max(model.sfc_list[sfc_index].tp,
                                                                      standby[standby_c2d[i - 1], standby_c2d[i]])
                for item in active:
                    if item not in standby.keys():
                        standby[item] = 0
                    if model.topo.edges[item]["active"] + active[item] + max(model.topo.edges[item]["reserved"],
                                                                             standby[item]) > \
                            model.topo.edges[item[0], item[1]]["bandwidth"]:
                        return False
                return True
            return True
        else:
            for item in active:
                if model.topo.edges[item[0], item[1]]["active"] + active[item] > model.topo.edges[item[0], item[1]][
                    "bandwidth"]:
                    return False
            return True

    if test_env != TestEnv.NoBackup:
        decision.set_active_path_s2c(paths[0][0])
        decision.set_active_path_c2d(paths[0][1])
        decision.set_standby_path_s2c(paths[1][0])
        decision.set_standby_path_c2d(paths[1][1])
        decision.set_update_path(paths[2])
    else:
        decision.set_active_path_s2c(paths[0])
        decision.set_active_path_c2d(paths[1])
    return True, decision


def verify_decision(model: Model, standby_verifier: Callable, active_index: int, standby_index: int, sfc_index: int,
                    test_env: TestEnv):
    """
    used to verify a decision

    :param model: model
    :param standby_verifier: standby verifier
    :param active_index: active server index
    :param standby_index: standby server index
    :param sfc_index: current sfc index
    :return: bool
    """
    if active_index == standby_index or not verify_active(model, sfc_index, active_index, test_env):
        return False

    if test_env == TestEnv.NoBackup or standby_verifier(model, sfc_index, active_index, standby_index, test_env):
        return True
    return False


def narrow_decision_set(model: Model, standby_verifier: Callable, sfc_index: int, test_env: TestEnv):
    """
    used to narrow available decision set

    the decision returned by this function is not fully evaluated for we just fill the active and
    standby rather than other fields such as path

    :param model: model
    :param standby_verifier: verify standby function
    :param sfc_index: cur processing sfc index
    :param test_env: test environment
    :return: [Decision]
    """
    decision_set = []

    # no backup condition
    if test_env == TestEnv.NoBackup:
        for i in range(len(model.topo.nodes)):
            if verify_active(model, sfc_index, i, test_env):
                decision_set.append(Decision(i, -1))

    # others
    for i in range(len(model.topo.nodes)):
        for j in range(len(model.topo.nodes)):
            if verify_decision(model, standby_verifier, i, j, sfc_index, test_env):
                decision_set.append(Decision(i, j))
    return decision_set


class RandomDecisionMaker(DecisionMaker):
    """
    this class used to make random decision
    """

    def __init__(self):
        super(RandomDecisionMaker, self).__init__()

    def generate_decision(self, model: Model, sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, needn't to check if it can be deployed

        :param model: model
        :param sfc_index: current sfc index
        :param state: state
        :param test_env: test environment
        :return: Decision
        """
        decision = Decision()
        decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
        decision.standby_server = -1 if test_env == TestEnv else random.sample(range(len(model.topo.nodes)), 1)[0]
        return decision


class RandomDecisionMakerWithGuarantee(DecisionMaker):
    """
    this class used to make random decision with guarantee
    """

    def __init__(self):
        super(RandomDecisionMakerWithGuarantee, self).__init__()

    def generate_decision(self, model: Model, sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, don't check if it can be deployed

        :param model: model
        :param sfc_index: current sfc index
        :param state: state
        :param test_env: test environment
        :return: Decision
        """
        decisions = narrow_decision_set(model, verify_standby_relax_flow_routing, sfc_index, test_env)
        if len(decisions) == 0:
            decision = Decision()
            decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
            decision.standby_server = -1 if test_env == TestEnv.NoBackup else \
                random.sample(range(len(model.topo.nodes)), 1)[0]
        else:
            decision = random.sample(decisions, 1)[0]
        return decision


class RandomDecisionMakerWithStrongGuarantee(DecisionMaker):
    """
    this class used to make random decision with strong guarantee
    """

    def __init__(self):
        super(RandomDecisionMakerWithStrongGuarantee, self).__init__()

    def generate_decision(self, model: Model, sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision

        :param model: model
        :param sfc_index: current sfc index
        :param state: state
        :param test_env: test environment
        :return: Decision
        """
        decisions = narrow_decision_set(model, verify_standby, sfc_index, test_env)

        if len(decisions) == 0:
            decision = Decision()
            decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
            decision.standby_server = -1 if test_env == TestEnv.NoBackup else \
                random.sample(range(len(model.topo.nodes)), 1)[0]
            return decision

        return random.sample(decisions, 1)[0]


def select_paths_for_updating(model: Model, sfc_index: int, active_index: int, standby_index: int, test_env: TestEnv):
    """
    select paths for determined active server index and standby server index

    :param model: model
    :param sfc_index: sfc index
    :param active_index: active server index
    :param standby_index: stand-by server index
    :param test_env: test environment
    :return: (bool, [])
    """
    # no backup condition
    assert test_env != TestEnv.NoBackup

    # calculate paths for updating
    update_paths = []
    temp_update = all_shortest_paths(model.topo, active_index, standby_index)[0]
    for path in all_shortest_paths(model.topo, active_index, standby_index):
        if is_path_throughput_met(model, path, model.sfc_list[sfc_index].update_tp, SFCType.Active, test_env):
            update_paths.append(path)
    if len(update_paths) == 0:
        return False, temp_update
    return True, random.sample(update_paths, 1)[0]


class ICCHeuristic(DecisionMaker):
    """
    this class used to make greedy decision of ICC paper raised
    """

    def __init__(self):
        super(ICCHeuristic, self).__init__()

    def generate_decision(self, model: Model, sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, don't check if it can be deployed

        :param model: model
        :param sfc_index: current sfc index
        :param state: state
        :param test_env: test environment
        :return: Decision
        """
        decisions = []

        # no backup condition
        if test_env == TestEnv.NoBackup:
            for i in range(len(model.topo.nodes)):
                decision = Decision()
                decision.active_server = i
                decision.standby_server = -1
                decision.flag = verify_decision(model, verify_standby_relax_flow_routing, i, -1, sfc_index, test_env)
                decisions.append(decision)

            decisions = sorted(decisions, key=lambda x: (model.topo.nodes[x.active_server]['computing_resource'] -
                                                         model.topo.nodes[x.active_server]['active']))
            for decision in decisions:
                if decision.flag:
                    return decision

            decision = Decision()
            decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
            decision.standby_server = -1
            return decision

        # others
        for i in range(len(model.topo.nodes)):
            for j in range(len(model.topo.nodes)):
                decision = Decision()
                decision.active_server = i
                decision.standby_server = j
                decision.flag = verify_decision(model, verify_standby_relax_flow_routing, i, j, sfc_index, test_env)
                decision.update_path = select_paths_for_updating(model, sfc_index, i, j, test_env)
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


class ICCDecisionMakerWithStrongGuarantee(DecisionMaker):
    """
    this class used to make random decision
    """

    def __init__(self):
        super(ICCDecisionMakerWithStrongGuarantee, self).__init__()

    def generate_decision(self, model: Model, sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, don't check if it can be deployed

        :param model: model
        :param sfc_index: current sfc index
        :param state: state
        :param test_env: test environment
        :return: Decision
        """
        decisions = []

        # no backup condition
        if test_env == TestEnv.NoBackup:
            for i in range(len(model.topo.nodes)):
                decision = Decision()
                decision.active_server = i
                decision.standby_server = -1
                decision.flag = verify_decision(model, verify_standby, i, -1, sfc_index, test_env)
                decisions.append(decision)

            decisions = sorted(decisions, key=lambda x: (model.topo.nodes[x.active_server]['computing_resource'] -
                                                         model.topo.nodes[x.active_server]['active']))
            for decision in decisions:
                if decision.flag:
                    return decision

            decision = Decision()
            decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
            decision.standby_server = -1
            return decision

        # others
        for i in range(len(model.topo.nodes)):
            for j in range(len(model.topo.nodes)):
                decision = Decision()
                decision.active_server = i
                decision.standby_server = j
                decision.flag = verify_decision(model, verify_standby, i, j, sfc_index, test_env)
                decision.update_path = select_paths_for_updating(model, sfc_index, i, j, test_env)
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
    this class used to make greedy decision of ICC paper raised
    """

    def __init__(self):
        super(Worst, self).__init__()

    def generate_decision(self, model: Model, sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, don't check if it can be deployed

        :param model: model
        :param sfc_index: current sfc index
        :param state: state
        :param test_env: test environment
        :return: Decision
        """
        decisions = []

        # no backup condition
        if test_env == TestEnv.NoBackup:
            for i in range(len(model.topo.nodes)):
                decision = Decision()
                decision.active_server = i
                decision.standby_server = -1
                decision.flag = verify_decision(model, verify_standby_relax_flow_routing, i, -1, sfc_index, test_env)
                decisions.append(decision)

            decisions = sorted(decisions, key=lambda x: (model.topo.nodes[x.active_server]['active'] -
                                                         model.topo.nodes[x.active_server]['computing_resource']))
            for decision in decisions:
                if decision.flag:
                    return decision

            decision = Decision()
            decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
            decision.standby_server = -1
            return decision

        # others
        for i in range(len(model.topo.nodes)):
            for j in range(len(model.topo.nodes)):
                decision = Decision()
                decision.active_server = i
                decision.standby_server = j
                decision.flag = verify_decision(model, verify_standby_relax_flow_routing, i, j, sfc_index, test_env)
                decision.update_path = select_paths_for_updating(model, sfc_index, i, j, test_env)
                decisions.append(decision)

        decisions = sorted(decisions, key=lambda x: (model.topo.nodes[x.active_server]['active'] -
                                                     model.topo.nodes[x.active_server]['computing_resource'],
                                                     len(x.update_path)))

        for decision in decisions:
            if decision.flag:
                return decision

        decision = Decision()
        decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
        decision.standby_server = random.sample(range(len(model.topo.nodes)), 1)[0]
        return decision


class WorstDecisionMakerWithStrongGuarantee(DecisionMaker):
    """
    this class used to make random decision
    """

    def __init__(self):
        super(WorstDecisionMakerWithStrongGuarantee, self).__init__()

    def generate_decision(self, model: Model, sfc_index: int, state: List, test_env: TestEnv):
        """
        generate new decision, don't check if it can be deployed

        :param model: model
        :param sfc_index: current sfc index
        :param state: state
        :param test_env: test environment
        :return: Decision
        """
        decisions = []

        # no backup condition
        if test_env == TestEnv.NoBackup:
            for i in range(len(model.topo.nodes)):
                decision = Decision()
                decision.active_server = i
                decision.standby_server = -1
                decision.flag = verify_decision(model, verify_standby, i, -1, sfc_index, test_env)
                decisions.append(decision)

            decisions = sorted(decisions, key=lambda x: (model.topo.nodes[x.active_server]['active'] -
                                                         model.topo.nodes[x.active_server]['computing_resource']))
            for decision in decisions:
                if decision.flag:
                    return decision

            decision = Decision()
            decision.active_server = random.sample(range(len(model.topo.nodes)), 1)[0]
            decision.standby_server = -1
            return decision

        # others
        for i in range(len(model.topo.nodes)):
            for j in range(len(model.topo.nodes)):
                decision = Decision()
                decision.active_server = i
                decision.standby_server = j
                decision.flag = verify_decision(model, verify_standby, i, j, sfc_index, test_env)
                decision.update_path = select_paths_for_updating(model, sfc_index, i, j, test_env)
                decisions.append(decision)

        decisions = sorted(decisions, key=lambda x: (model.topo.nodes[x.active_server]['active'] -
                                                     model.topo.nodes[x.active_server]['computing_resource'],
                                                     len(x.update_path)))

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
