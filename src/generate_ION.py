from generate_topo import *

if __name__ == "__main__":
    topo = nx.readwrite.gml.read_gml("dataset\\Ion.gml", label=None)
    cs_low = 60000
    cs_high = 80000
    bandwidth_low = 400
    bandwidth_high = 800
    fail_rate_low = 0.0
    fail_rate_high = 0.4
    latency_low = 2
    latency_high = 5

    # generate V
    for i in range(125):
        topo.nodes[i]['computing_resource'] = random.randint(cs_low, cs_high)
        topo.nodes[i]['fail_rate'] = random.uniform(fail_rate_low, fail_rate_high)
        topo.nodes[i]['active'] = 0
        topo.nodes[i]['reserved'] = 0
        topo.nodes[i]['max_sbsfc_index'] = -1
        topo.nodes[i]['sbsfcs'] = set()

    for src, dst, _ in topo.edges(data=True):
        bandwidth = random.randint(bandwidth_low, bandwidth_high)
        latency = random.uniform(latency_low, latency_high)
        topo.add_edge(src, dst, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1,
                      sbsfcs_s2c=set(), sbsfcs_c2d=set())

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

