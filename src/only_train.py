from tqdm import tqdm
import torch.optim as optim
from generate_topo import *
from train_dqn import GAMMA, EPSILON_DECAY, EPSILON_FINAL, EPSILON_START, EPSILON, ACTION_SPACE, DEVICE, ACTION_LEN, DOUBLE

# parameters with rl
if pf == "Windows":
    SAMPLE_FILE = "model\\sample"
    TARGET_FILE = "model\\target"
    EXP_REPLAY_FILE = "model\\replay.pkl"
elif pf == "Linux":
    SAMPLE_FILE = "model/sample"
    TARGET_FILE = "model/target"
    EXP_REPLAY_FILE = "model/replay.pkl"

BATCH_SIZE = 32 # start with small（32）, then go to big

LEARNING_RATE = 1e-3
SYNC_INTERVAL = 1
TRAIN_INTERVAL = 1
TRAIN_ITERATIONS = 10000

if __name__ == "__main__":
    # create model
    if load_model:
        with open(model_file_name, 'rb') as f:
            model = pickle.load(f)  # read file and build object
            STATE_LEN = len(model.topo.nodes()) * 4 + len(model.topo.edges()) * 4 + 7
    else:
        with open(topo_file_name, 'rb') as f:
            topo = pickle.load(f)  # read file and build object
            sfc_list = generate_sfc_list(topo, process_capacity, size=sfc_size, duration=duration, jitter=jitter)
            model = Model(topo=topo, sfc_list=sfc_list)
            STATE_LEN = len(topo.nodes()) * 4 + len(topo.edges()) * 4 + 7

    # create decision maker(agent) & optimizer & environment
    # create net and target net
    LEARNING_FROM_LAST = True if os.path.exists(TARGET_FILE) and os.path.exists(SAMPLE_FILE) else False

    # create decision maker(agent) & optimizer & environment
    # create net and target net
    if LEARNING_FROM_LAST:
        net = torch.load(SAMPLE_FILE)
        tgt_net = torch.load(TARGET_FILE)
        # buffer = ExperienceBuffer(capacity=REPLAY_SIZE)
        with open(EXP_REPLAY_FILE, 'rb') as f:
            buffer = pickle.load(f)  # read file and build object
    else:
        net = DQN(state_len=STATE_LEN, action_len=ACTION_LEN, device=DEVICE, tgt=False)
        tgt_net = DQN(state_len=STATE_LEN, action_len=ACTION_LEN, device=DEVICE, tgt=True)
        for target_param, param in zip(tgt_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data)
        with open(EXP_REPLAY_FILE, 'rb') as f:
            buffer = pickle.load(f)  # read file and build object

    decision_maker = DQNDecisionMaker(net=net, tgt_net=tgt_net, buffer=buffer, action_space=ACTION_SPACE, epsilon=EPSILON, epsilon_start=EPSILON_START, epsilon_final=EPSILON_FINAL, epsilon_decay=EPSILON_DECAY, device=DEVICE, gamma=GAMMA)

    optimizer = optim.Adam(decision_maker.net.parameters(), lr=LEARNING_RATE)

    for it in tqdm(range(TRAIN_ITERATIONS)):
        optimizer.zero_grad()
        batch = decision_maker.buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, decision_maker.net, decision_maker.tgt_net, gamma=decision_maker.gamma, nodes_number=topo_size, double=DOUBLE, device=DEVICE)
        loss_t.backward()
        optimizer.step()
        decision_maker.tgt_net.load_state_dict(decision_maker.net.state_dict())

    torch.save(decision_maker.net, SAMPLE_FILE)
    torch.save(decision_maker.tgt_net, TARGET_FILE)
    with open(EXP_REPLAY_FILE, 'wb') as f:  # open file with write-mode
        model_string = pickle.dump(decision_maker.buffer, f)  # serialize and save object

        # Monitor.print_log()
        # model.print_start_and_down()

    print("done!")
