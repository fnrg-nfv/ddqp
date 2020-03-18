from generate_topo import *

if __name__ == "__main__":
    topo = nx.Graph()
    cs_low = 20000
    cs_high = 40000
    bandwidth_low = 200
    bandwidth_high = 400
    fail_rate_low = 0.0
    fail_rate_high = 0.4
    latency_low = 2
    latency_high = 5

    # generate V
    for i in range(9):
        computing_resource = random.randint(cs_low, cs_high)
        fail_rate = random.uniform(fail_rate_low, fail_rate_high)
        topo.add_node(i, computing_resource=computing_resource, fail_rate=fail_rate, active=0, reserved=0, max_sbsfc_index=-1, sbsfcs=set())

    # generate E
    for i in range(1, 9):
        bandwidth = random.randint(bandwidth_low, bandwidth_high)
        latency = random.uniform(latency_low, latency_high)
        topo.add_edge(0, i, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())

    for i in range(2, 9):
        bandwidth = random.randint(bandwidth_low, bandwidth_high)
        latency = random.uniform(latency_low, latency_high)
        topo.add_edge(1, i, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())

    for i in range(3, 9):
        bandwidth = random.randint(bandwidth_low, bandwidth_high)
        latency = random.uniform(latency_low, latency_high)
        topo.add_edge(2, i, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())

    for i in range(4, 9):
        bandwidth = random.randint(bandwidth_low, bandwidth_high)
        latency = random.uniform(latency_low, latency_high)
        topo.add_edge(3, i, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())

    for i in range(5, 9):
        bandwidth = random.randint(bandwidth_low, bandwidth_high)
        latency = random.uniform(latency_low, latency_high)
        topo.add_edge(4, i, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())

    for i in range(6, 9):
        bandwidth = random.randint(bandwidth_low, bandwidth_high)
        latency = random.uniform(latency_low, latency_high)
        topo.add_edge(5, i, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())

    for i in range(7, 9):
        bandwidth = random.randint(bandwidth_low, bandwidth_high)
        latency = random.uniform(latency_low, latency_high)
        topo.add_edge(6, i, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())

    for i in range(8, 9):
        bandwidth = random.randint(bandwidth_low, bandwidth_high)
        latency = random.uniform(latency_low, latency_high)
        topo.add_edge(7, i, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())

    if load_model:
        with open(model_file_name, 'wb') as f:  # open file with write-mode
            sfc_list = generate_sfc_list(topo, process_capacity, size=sfc_size, duration=duration, jitter=jitter)
            nx.draw(topo)
            show()
            model = Model(topo=topo, sfc_list=sfc_list)
            model_string = pickle.dump(model, f)  # serialize and save object
    else:
        with open(topo_file_name, 'wb') as f:  # open file with write-mode
            nx.draw(topo)
            show()
            topo_string = pickle.dump(topo, f)  # serialize and save object

