from tqdm import tqdm
import torch.optim as optim
from generate_topo import *

# parameters with rl
if pf == "Windows":
    SAMPLE_FILE = "model\\sample"
    TARGET_FILE = "model\\target"
    EXP_REPLAY_FILE = "model\\replay.pkl"
    TRACE_FILE = "model\\trace.pkl"
elif pf == "Linux":
    SAMPLE_FILE = "model/sample"
    TARGET_FILE = "model/target"
    EXP_REPLAY_FILE = "model/replay.pkl"
    TRACE_FILE = "model/trace.pkl"
else:
    raise RuntimeError('Platform unsupported')

GAMMA = 0
BATCH_SIZE = 32 # start with small（32）, then go to big

ACTION_SHAPE = 2
REPLAY_SIZE = 10000
EPSILON = 0.0
EPSILON_START = 1.0
EPSILON_FINAL = 0.3
EPSILON_DECAY = 50
LEARNING_RATE = 1e-3
SYNC_INTERVAL = 500
TRAIN_INTERVAL = 1
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")
ITERATIONS = 10000
DOUBLE = True
TEST = True

if load_model:
    with open(model_file_name, 'rb') as f:
        model = pickle.load(f)  # read file and build object
        STATE_LEN = len(model.topo.nodes()) * 3 + len(model.topo.edges()) * 3 + 7
else:
    with open(topo_file_name, 'rb') as f:
        topo = pickle.load(f)  # read file and build object
        STATE_LEN = len(topo.nodes()) * 3 + len(topo.edges()) * 3 + 7

if __name__ == "__main__":
    for it in range(ITERATIONS):

        # create model
        if load_model:
            with open(model_file_name, 'rb') as f:
                model = pickle.load(f)  # read file and build object
        else:
            with open(topo_file_name, 'rb') as f:
                topo = pickle.load(f)  # read file and build object
                sfc_list = generate_sfc_list(topo, process_capacity, size=sfc_size, duration=duration, jitter=jitter)
                model = Model(topo=topo, sfc_list=sfc_list)

        LEARNING_FROM_LAST = True if os.path.exists(TARGET_FILE) and os.path.exists(SAMPLE_FILE) and os.path.exists(EXP_REPLAY_FILE) else False

        # create decision maker(agent) & optimizer & environment
        # create net and target net
        if LEARNING_FROM_LAST:
            net = torch.load(SAMPLE_FILE)
            tgt_net = torch.load(TARGET_FILE)
            # buffer = ExperienceBuffer(capacity=REPLAY_SIZE)
            with open(EXP_REPLAY_FILE, 'rb') as f:
                buffer = pickle.load(f)  # read file and build object
        else:
            net = BranchingQNetwork(state_len=STATE_LEN, dimensions=2, actions_per_dimension=len(model.topo.nodes()), is_tgt=False, is_fc=True,  device=DEVICE)
            tgt_net = BranchingQNetwork(state_len=STATE_LEN, dimensions=2, actions_per_dimension=len(model.topo.nodes()),
                                    is_tgt=True, is_fc=True, device=DEVICE)
            for target_param, param in zip(tgt_net.parameters(), net.parameters()):
                target_param.data.copy_(param.data)
            buffer = PrioritizedExperienceBuffer(capacity=REPLAY_SIZE, alpha=1)
        if os.path.exists(TRACE_FILE):
            with open(TRACE_FILE, 'rb') as f:
                reward_trace = pickle.load(f)  # read file and build object
        else:
            reward_trace = []

        decision_maker = BranchingDecisionMaker(net=net, tgt_net=tgt_net, buffer=buffer, gamma=GAMMA, epsilon_start=EPSILON_START, epsilon=EPSILON, epsilon_final=EPSILON_FINAL, epsilon_decay=EPSILON_DECAY, model=model, device=DEVICE)

        optimizer = optim.Adam(decision_maker.net.parameters(), lr=LEARNING_RATE)
        # optimizer = optim.SGD(decision_maker.net.parameters(), lr=LEARNING_RATE, momentum=0.9)
        env = DQNEnvironment()

        # related
        action = VariableState.Uninitialized
        reward = VariableState.Uninitialized
        state = VariableState.Uninitialized
        idx = 0

        # main function
        for cur_time in tqdm(range(0, duration)):

            # generate failed instances
            failed_instances = generate_failed_instances_time_slot(model, cur_time)
            # failed_instances = []
            # handle state transition
            state_transition_and_resource_reclaim(model, cur_time, test_env, failed_instances)

            # process all sfcs
            for i in range(len(model.sfc_list)):
                # for each sfc which locate in this time slot
                if cur_time <= model.sfc_list[i].time < cur_time + 1:
                    idx += 1
                    state, _ = env.get_state(model=model, sfc_index=i)
                    decision = deploy_sfc_item(model, i, decision_maker, cur_time, state, test_env)
                    action = DQNAction(decision.active_server, decision.standby_server).get_action()
                    reward = env.get_reward(model, i)
                    next_state, done = env.get_state(model=model, sfc_index=i + 1)

                    exp = Experience(state=state, action=action, reward=reward, done=done, new_state=next_state)
                    decision_maker.append_sample(exp, GAMMA)

                    if len(decision_maker.buffer) < REPLAY_SIZE:
                        continue

                    if idx % SYNC_INTERVAL == 0:
                        decision_maker.tgt_net.load_state_dict(decision_maker.net.state_dict())

                    if idx % TRAIN_INTERVAL == 0:
                        optimizer.zero_grad()
                        batch, indices, importances = decision_maker.buffer.sample(BATCH_SIZE)
                        loss_t = calc_loss_branching_prio(batch, decision_maker.net, decision_maker.tgt_net, GAMMA, indices,
                                                          importances,
                                                          decision_maker.buffer, device=DEVICE)
                        loss_t.backward()
                        # print(decision_maker.net.fc7.weight.data)
                        optimizer.step()
        torch.save(decision_maker.net, SAMPLE_FILE)
        torch.save(decision_maker.tgt_net, TARGET_FILE)
        with open(EXP_REPLAY_FILE, 'wb') as f:  # open file with write-mode
            model_string = pickle.dump(decision_maker.buffer, f)  # serialize and save object

        # test
        if TEST:
            action_list = []
            tgt_net = decision_maker.tgt_net
            buffer = PrioritizedExperienceBuffer(capacity=REPLAY_SIZE, alpha=1)
            decision_maker = BranchingDecisionMaker(net=net, tgt_net=tgt_net, buffer=buffer, gamma=GAMMA,
                                                    epsilon_start=EPSILON_START, epsilon=EPSILON,
                                                    epsilon_final=EPSILON_FINAL, epsilon_decay=EPSILON_DECAY,
                                                    model=model, device=DEVICE)

            if load_model:
                with open(model_file_name, 'rb') as f:
                    model = pickle.load(f)  # read file and build object
            else:
                with open(topo_file_name, 'rb') as f:
                    topo = pickle.load(f)  # read file and build object
                    sfc_list = generate_sfc_list(topo=topo, process_capacity=process_capacity, size=sfc_size, duration=duration, jitter=jitter)
                    model = Model(topo, sfc_list)
            STATE_SHAPE = (len(model.topo.nodes()) + len(model.topo.edges())) * 3 + 7

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
            # plot_action_distribution(action_list, num_nodes=topo_size)
            total_reward = model.calculate_total_reward()
            reward_trace.append(total_reward)

        with open(TRACE_FILE, 'wb') as f:  # open file with write-mode
            pickle.dump(reward_trace, f)  # serialize and save object

        # Monitor.print_log()
        # model.print_start_and_down()
        print("iteration: ", it)
        report(model)
