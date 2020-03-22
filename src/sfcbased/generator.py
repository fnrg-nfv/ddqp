import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
import numpy as np
from sfcbased.model import *

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
fig, ax = plt.subplots()
fig.set_tight_layout(False)


def generate_action_space(size: int):
    """
    generate action space which contains all actions

    :param size: number of nodes
    :return: [(int, int)] action space
    """
    return [(i, j) for i in range(size) for j in range(size)]


def generate_topology(size: int = 100):
    """
    function used to generate topology

    mainly with three resources: computing resources, bandwidth resources and latency resources
    notices:
    1. active: double the resources occupied by active instance
    2. reserved: double the resources reserved by standby instance
    3. max_sbsfc_index: int the index of standby sfc which has largest reservation, only for MaxReservation
    4. sbsfcs: [int] the index of standby instance deployed on this server(not started)

    :param size: node numbers
    :return: nx.Graph topology
    """
    topo = nx.Graph()
    cs_low = 20000
    cs_high = 40000
    bandwidth_low = 100
    bandwidth_high = 300
    fail_rate_low = 0.0
    fail_rate_high = 0.4
    inconnectivity = 15
    latency_low = 2
    latency_high = 5

    # generate V
    for i in range(size):
        computing_resource = random.randint(cs_low, cs_high)
        fail_rate = random.uniform(fail_rate_low, fail_rate_high)
        topo.add_node(i, computing_resource=computing_resource, fail_rate=fail_rate, active=0, reserved=0, max_sbsfc_index=-1, sbsfcs=set())

    # generate E
    for i in range(size):
        for j in range(i + 1, size):
            # make sure the whole network is connected
            if j == i + 1:
                bandwidth = random.randint(bandwidth_low, bandwidth_high)
                topo.add_edge(i, j, bandwidth=bandwidth, active=0, reserved=0, latency=random.uniform(latency_low, latency_high), max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
                continue
            if random.randint(1, inconnectivity) == 1:
                bandwidth = random.randint(bandwidth_low, bandwidth_high)
                topo.add_edge(i, j, bandwidth=bandwidth, active=0, reserved=0, latency=random.uniform(latency_low, latency_high), max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    return topo


def generate_sfc_list(topo: nx.Graph, process_capacity: int, size: int = 100, duration: int = 100, jitter: bool = True):
    """
    generate specified number SFCs

    :param topo: network topology(used to determine the start server and the destination server of specified SFC)
    :param process_capacity: process capacity in each time slot
    :param size: the total number SFCs(not exactly)
    :param duration: arriving SFCs duration
    :param jitter: if sfc
    :return: [SFC]
    """
    sfc_list = []
    nodes_len = len(topo.nodes)
    timeslot_list = [random.uniform(0, duration) for _ in range(size)]
    timeslot_list.sort()

    # generate each sfc
    cs_low =  3750
    cs_high = 7500
    tp_low = 32 # 32
    tp_high = 128 # 128
    latency_low = 10 # 10
    latency_high = 30 # 30
    process_latency_low = 0.863
    process_latency_high = 1.725
    TTL_low = 5
    TTL_high = 10

    if jitter:
        for i in range(len(timeslot_list)):
            computing_resource = random.uniform(cs_low, cs_high) # 5625
            tp = random.uniform(tp_low, tp_high) # 80
            latency = random.uniform(latency_low, latency_high) # 20
            update_tp = 0.001 * computing_resource
            process_latency = random.uniform(process_latency_low, process_latency_high) # 1.294
            TTL = random.randint(TTL_low, TTL_high)  # sfc's time to live
            s = random.randint(1, nodes_len - 1)
            d = random.randint(1, nodes_len - 1)
            sfc_list.append(SFC(computing_resource, tp, latency, update_tp, process_latency, s, d, timeslot_list[i], TTL))
    else:
        computing_resource = (cs_low + cs_high) / 2  # 5625
        tp = (tp_low + tp_high) / 2  # 80
        latency = (latency_low + latency_high) / 2  # 20
        update_tp = 0.001 * computing_resource
        process_latency = (process_latency_low + process_latency_high) / 2  # 1.294
        for i in range(len(timeslot_list)):
            TTL = random.randint(TTL_low, TTL_high)  # sfc's time to live
            s = random.randint(1, nodes_len - 1)
            d = random.randint(1, nodes_len - 1)
            sfc_list.append(SFC(computing_resource, tp, latency, update_tp, process_latency, s, d, timeslot_list[i], TTL))

    return sfc_list


def generate_model(topo_size: int = 100, sfc_size: int = 100, duration: int = 100, process_capacity: int = 10):
    """
    function used to generate specified number nodes in network topology and SFCs in SFC list

    :param topo_size: nodes number in network topology
    :param sfc_size: SFCs number in SFC list
    :param duration: Duration of model
    :return: Model model
    """
    topo = generate_topology(size=topo_size)
    sfc_list = generate_sfc_list(topo=topo, size=sfc_size, duration=duration, process_capacity=process_capacity)
    return Model(topo, sfc_list)


def generate_failed_instances_time_slot(model: Model, time: int):
    """
    random generate failed instances

    random generate failed instances which are:
    1. either active or stand-by instance is running
    2. can't expired in this time slot

    :param model: model
    :param time: current time
    :return: [Instance] list of instance
    """
    nodes = model.topo.nodes(data=True)
    down_servers = [i for i in range(len(nodes)) if np.random.random() < nodes[i]['fail_rate']]
    instances = []
    for i in range(len(model.sfc_list)):
        cur_sfc = model.sfc_list[i]
        if cur_sfc.state == State.Normal and cur_sfc.time + cur_sfc.TTL >= time and cur_sfc.active_sfc.server in down_servers:
            instances.append(Instance(i, True))
        if cur_sfc.state == State.Backup and cur_sfc.time + cur_sfc.TTL >= time and cur_sfc.standby_sfc.server in down_servers:
            instances.append(Instance(i, False))
    return tuple(instances)

    # random fail
    # assert error_rate <= 1
    #
    # # get all running instances
    # all_running_instances = []
    # for i in range(len(model.sfc_list)):
    #     cur_sfc = model.sfc_list[i]
    #     if cur_sfc.state == State.Normal and cur_sfc.time + cur_sfc.TTL >= time:
    #         all_running_instances.append(Instance(i, True))
    #     if model.sfc_list[i].state == State.Backup and cur_sfc.time + cur_sfc.TTL >= time:
    #         all_running_instances.append(Instance(i, False))
    #
    # # random select
    # sample_num = math.ceil(len(all_running_instances) * error_rate)
    # failed_instances = random.sample(all_running_instances, sample_num)
    # return failed_instances


# test
def __main():
    topo = generate_topology()
    print("Num of edges: ", len(topo.edges))
    print("Edges: ", topo.edges.data())
    print("Nodes: ", topo.nodes.data())
    print(topo[0])
    nx.draw(topo, with_labels=True)
    plt.show()


if __name__ == '__main__':
    __main()
