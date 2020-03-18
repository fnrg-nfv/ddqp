from vnfbased import *

# meta-parameters
time_step = 1 # time slot
topo_size = 10 # topology size
sfc_size = 20 # number of SFCs
duration = 100 # simulation time

model = generate_model(topo_size=topo_size, sfc_size=sfc_size, duration=duration)

decision_maker = RandomDecisionMaker()

# nx.draw(model.topo, with_labels=True)
# plt.show()

for cur_time in range(0, duration, time_step):
    process_time_slot(model, decision_maker, cur_time, time_step)

Monitor.print_log()

print("\nDone!")


