from generate_topo import *

if __name__ == "__main__":
    if load_model:
        with open(model_file_name, 'rb') as f:
            model = pickle.load(f)  # read file and build object
            topo = model.topo
    else:
        with open(topo_file_name, 'rb') as f:
            topo = pickle.load(f)  # read file and build object

    print("\nservers: ", len(topo.nodes()))
    print("edges:", len(topo.edges()))
    for node in topo.nodes(data=True):
        print(node[1]['computing_resource'])
    for edge in topo.edges(data=True):
        print(edge[2]['bandwidth'])
        print(edge[2]['latency'])

    nx.draw(topo)
    show()