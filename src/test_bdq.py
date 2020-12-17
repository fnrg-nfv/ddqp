from tqdm import tqdm
from generate_topo import *
from train_dqn import REPLAY_SIZE, EPSILON, EPSILON_START, EPSILON_FINAL, EPSILON_DECAY, GAMMA, STATE_LEN, ACTION_LEN, ACTION_SPACE, TARGET_FILE

# parameters with rl
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#DEVICE = torch.device("cpu")
ITERATIONS = 5

# create decision maker(agent) & optimizer & environment
# create net and target net
tgt_net = torch.load(TARGET_FILE)
buffer = ExperienceBuffer(capacity=REPLAY_SIZE)

# decision_maker = DQNDecisionMaker(net=tgt_net, tgt_net = tgt_net, buffer = buffer, action_space = ACTION_SPACE, epsilon = EPSILON, epsilon_start = EPSILON_START, epsilon_final = EPSILON_FINAL, epsilon_decay = EPSILON_DECAY, device = DEVICE, gamma = GAMMA, model=model)

env = DQNEnvironment()

# related
action = VariableState.Uninitialized
reward = VariableState.Uninitialized
state = VariableState.Uninitialized
idx = 0

# main function
if __name__ == "__main__":
    fail_rates = []
    throughputs = []
    service_times = []
    total_rewards = []
    accept_rates = []
    accept_nums = []
    real_fail_rate = 0
    for it in range(0, ITERATIONS):
        action_list = []
        if load_model:
            with open(model_file_name, 'rb') as f:
                model = pickle.load(f)  # read file and build object
        else:
            with open(topo_file_name, 'rb') as f:
                topo = pickle.load(f)  # read file and build object
                sfc_list = generate_sfc_list(topo=topo, process_capacity=process_capacity, size=sfc_size, duration=duration, jitter=jitter)
                model = Model(topo, sfc_list)
        STATE_SHAPE = (len(model.topo.nodes()) + len(model.topo.edges())) * 3 + 7
        decision_maker = BranchingDecisionMaker(net=tgt_net, tgt_net=tgt_net, buffer=buffer, gamma=GAMMA,
                                                epsilon_start=EPSILON_START, epsilon=EPSILON,
                                                epsilon_final=EPSILON_FINAL, epsilon_decay=EPSILON_DECAY,
                                                model=model, device=DEVICE)

        for cur_time in tqdm(range(0, duration)):

            # generate failed instances
            failed_instances = generate_failed_instances_time_slot(model, cur_time)
            # failed_instances = []
            # handle state transition
            state_transition_and_resource_reclaim(model, cur_time, test_env, failed_instances)

            # deploy sfcs / handle each time slot
            for i in range(len(model.sfc_list)):
                # for each sfc which locate in this time slot
                if cur_time <= model.sfc_list[i].time < cur_time + 1:
                    idx += 1
                    state, _ = env.get_state(model=model, sfc_index=i)
                    decision = deploy_sfc_item(model, i, decision_maker, cur_time, state, test_env)
                    action = DQNAction(decision.active_server, decision.standby_server).get_action()
                    action_list.append(action)

        # Monitor.print_log()
        # model.print_start_and_down()
        plot_action_distribution(action_list, num_nodes=topo_size)

        fail_rate, real_fail_rate, throughput, service_time, total_reward, accept_num, place_num, accept_rate, place_cdf = report(model)
        fail_rates.append(fail_rate)
        throughputs.append(throughput)
        service_times.append(service_time)
        total_rewards.append(total_reward)
        accept_rates.append(accept_rate)
        accept_nums.append(accept_num)
    print("avg fail rate: ", sum(fail_rates)/len(fail_rates))
    print("avg real fail rate: ", real_fail_rate)
    print("avg throughput: ", sum(throughputs)/len(throughputs))
    print("avg service time: ", sum(service_times)/len(service_times))
    print("avg total reward: ", sum(total_rewards)/len(total_rewards))
    print("avg accept nums: ", sum(accept_nums) / len(accept_nums))
    print("avg accept rate: ", sum(accept_rates)/len(accept_rates))