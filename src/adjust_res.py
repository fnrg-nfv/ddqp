from generate_topo import topo_file_name, model_file_name, load_model, pickle, random

cs_low = 60000 #10000
cs_high = 80000 #20000
bandwidth_low = 400 # 1000
bandwidth_high = 800 # 3000
fail_rate_low = 0.0
fail_rate_high = 0.4
latency_low = 2 # 2
latency_high = 5 # 5

if __name__ == "__main__":
    if load_model:
        with open(model_file_name, 'rb') as f:
            model = pickle.load(f)  # read file and build object
            topo = model.topo
        with open(model_file_name, 'wb') as f:
            for node in topo.nodes(data=True):
                computing_resource = random.randint(cs_low, cs_high)
                fail_rate = random.uniform(fail_rate_low, fail_rate_high)
                node[1]["computing_resource"] = computing_resource
                node[1]["fail_rate"] = fail_rate

            for edge in topo.edges(data=True):
                bandwidth = random.randint(bandwidth_low, bandwidth_high)
                latency = random.uniform(latency_low, latency_high)
                edge[2]['bandwidth'] = bandwidth
                edge[2]['latency'] = latency

            model_string = pickle.dump(model, f)  # serialize and save object
    else:
        with open(topo_file_name, 'rb') as f:  # open file with write-mode
            topo = pickle.load(f)  # read file and build object
        with open(topo_file_name, 'wb') as f:  # open file with write-mode
            for node in topo.nodes(data=True):
                computing_resource = random.randint(cs_low, cs_high)
                fail_rate = random.uniform(fail_rate_low, fail_rate_high)
                node[1]["computing_resource"] = computing_resource
                node[1]["fail_rate"] = fail_rate

            for edge in topo.edges(data=True):
                bandwidth = random.randint(bandwidth_low, bandwidth_high)
                latency = random.uniform(latency_low, latency_high)
                edge[2]['bandwidth'] = bandwidth
                edge[2]['latency'] = latency

            topo_string = pickle.dump(topo, f)  # serialize and save object

