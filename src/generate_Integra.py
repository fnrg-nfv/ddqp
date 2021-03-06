from generate_topo import *

if __name__ == "__main__":
    topo = nx.Graph()
    cs_low = 60000
    cs_high = 80000
    bandwidth_low = 400
    bandwidth_high = 800
    fail_rate_low = 0.0
    fail_rate_high = 0.4
    latency_low = 2
    latency_high = 5

    # generate V
    for i in range(27):
        computing_resource = random.randint(cs_low, cs_high)
        fail_rate = random.uniform(fail_rate_low, fail_rate_high)
        topo.add_node(i, computing_resource=computing_resource, fail_rate=fail_rate, active=0, reserved=0, max_sbsfc_index=-1, sbsfcs=set())

    # generate E
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(0, 16, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(0, 3, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(0, 20, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(1, 20, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(1, 7, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(2, 18, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(2, 3, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(3, 18, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(3, 26, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(3, 15, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(4, 5, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(4, 6, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(4, 7, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(5, 8, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(6, 23, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(6, 7, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(8, 9, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(8, 13, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(8, 14, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(8, 25, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(10, 25, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(10, 12, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(11, 25, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(11, 26, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(13, 23, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(15, 16, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(17, 26, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(17, 19, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(19, 20, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(20, 21, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(21, 22, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(21, 23, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(22, 23, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(23, 24, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(24, 25, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(25, 26, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())

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

