from tqdm import tqdm
from generate_topo import *

if load_model:
    with open(model_file_name, 'rb') as f:
        model = pickle.load(f)  # read file and build object
else:
    with open(topo_file_name, 'rb') as f:
        topo = pickle.load(f)  # read file and build object
        sfc_list = generate_sfc_list(topo=topo, process_capacity=process_capacity, size=sfc_size, duration=duration, jitter=jitter)
        model = Model(topo, sfc_list)

decision_maker = RandomDecisionMakerWithStrongGuarantee()

# nx.draw(model.topo, with_labels=True)
# plt.show()

env = NormalEnvironment()

for cur_time in tqdm(range(0, duration)):
    failed_instances = generate_failed_instances_time_slot(model, cur_time)
    # failed_instances = []
    state = env.get_state(model, 0)
    process_time_slot(model, decision_maker, cur_time, test_env, state, failed_instances)

# Monitor.print_log()

# model.print_start_and_down()
report(model)

print("\nDone!")


