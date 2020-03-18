import matplotlib.pyplot as plt
from generate_topo import *
import matplotlib as mpl
from matplotlib.lines import Line2D

# Vertical
# Acceptance rate, Throughput, Service time, Running time cost, Average total reward
# Acceptance rate, Real fail rate, Service time

# Acceptance rate between algorithms
# NOSD, DDQP, BFG, RG

path_prefix = "..\\result\\" if pf == "Windows" else "../result/"  # file name

dpi = 100
fontsize = 14
hvretio = 7.5


def plot_acceptance_rate_between_algorithms():
    Abilene = [0.4320987654320988, 0.4752908229578495, 0.39800332778702163, 0.3138806462248599]
    ANS = [0.4923941798941799, 0.5054905479205356, 0.40666439986399183, 0.3404401650618982]
    BSO = [0.47676008202323994, 0.4876654042272399, 0.39457735247208936, 0.3543358233192595]
    AboveNet = [0.47622329427980703, 0.46996842474409045, 0.4270145310435931, 0.3010288748755393]
    Integra = [0.4881944444444445, 0.4669103518061716, 0.4333223575897267, 0.27026143790849677]
    BICS = [0.5595643294758339, 0.5485782810183799, 0.4693946567467028, 0.3015710382513661]

    plt.ylim(0.75, 2)

    higher_BFG = (Abilene[1] / Abilene[2] + ANS[1] / ANS[2] + AboveNet[1] / AboveNet[2]
                  + Integra[1] / Integra[2] + BICS[1] / BICS[2]) / 5 - 1
    higher_RG = (Abilene[1] / Abilene[3] + ANS[1] / ANS[3] + AboveNet[1] / AboveNet[3]
                 + Integra[1] / Integra[3] + BICS[1] / BICS[3]) / 5 - 1
    higher_NOSD = (Abilene[1] / Abilene[0] + ANS[1] / ANS[0] + AboveNet[1] / AboveNet[0]
                   + Integra[1] / Integra[0] + BICS[1] / BICS[0]) / 5 - 1
    print("acceptance_rate: avg higher than RG: {}, avg higher than BFG: {}, avg higher than NOSD: {}".format(higher_RG,
                                                                                                              higher_BFG,
                                                                                                              higher_NOSD))
    plt.ylabel('Acceptance Rate/RG', fontsize=fontsize)

    labels = ["Abilene", "ANS", "AboveNet", "Integra", "BICS"]

    bfg_index = [0.4, 1.4, 2.4, 3.4, 4.4]
    nosd_index = [0.6, 1.6, 2.6, 3.6, 4.6]
    ddqp_index = [0.8, 1.8, 2.8, 3.8, 4.8]
    plt.xlim(0, 5.2)

    label_index = [0.6, 1.6, 2.6, 3.6, 4.6]
    width = 0.16

    plt.bar(bfg_index, [Abilene[2] / Abilene[3], ANS[2] / ANS[3], AboveNet[2] / AboveNet[3], Integra[2] / Integra[3], BICS[2] / BICS[3]], width, color="#0d7263", label="BFG",
            lw=1, edgecolor="black")
    plt.bar(nosd_index, [Abilene[0] / Abilene[3], ANS[0] / ANS[3], AboveNet[0] / AboveNet[3], Integra[0] / Integra[3], BICS[0] / BICS[3]], width, color='#ef8935', label="NOSD",
            lw=1, edgecolor="black")
    plt.bar(ddqp_index, [Abilene[1] / Abilene[3], ANS[1] / ANS[3], AboveNet[1] / AboveNet[3], Integra[1] / Integra[3], BICS[1] / BICS[3]], width, color='#992020', label="DDQP",
            lw=1, edgecolor="black")
    plt.hlines(1.0, 0, 5.2, colors="#132486", lw = 2, linestyles="--")

    plt.xticks(label_index, labels=labels, fontsize=fontsize)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend(fontsize=15)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(hvretio, 5)
    fig.savefig(path_prefix + 'acceptance.png', dpi=dpi)
    plt.show()


def plot_throughput_between_algorithms():
    Abilene = [97105, 105089.0, 88819, 69440]
    ANS = [109368, 113281.9, 91110, 74558]
    BSO = [105891, 109145.3, 93238, 81377]
    AboveNet = [103858, 105863.9, 95085, 69707]
    Integra = [106604, 105876.4, 98191, 63350]
    BICS = [127957, 125447.4, 105947, 67995]
    plt.ylim(0.75, 2)
    higher_BFG = (Abilene[1] / Abilene[2] + ANS[1] / ANS[2] + AboveNet[1] / AboveNet[2]
                  + Integra[1] / Integra[2] + BICS[1] / BICS[2]) / 5 - 1
    higher_RG = (Abilene[1] / Abilene[3] + ANS[1] / ANS[3] + AboveNet[1] / AboveNet[3]
                 + Integra[1] / Integra[3] + BICS[1] / BICS[3]) / 5 - 1
    higher_NOSD = (Abilene[1] / Abilene[0] + ANS[1] / ANS[0] + AboveNet[1] / AboveNet[0]
                   + Integra[1] / Integra[0] + BICS[1] / BICS[0]) / 5 - 1

    print("throughput: avg higher than RG: {}, avg higher than BFG: {}, avg higher than NOSD: {}".format(higher_RG,
                                                                                                         higher_BFG,
                                                                                                         higher_NOSD))

    plt.ylabel('Throughput/RG', fontsize=fontsize)

    labels = ["Abilene", "ANS", "AboveNet", "Integra", "BICS"]

    bfg_index = [0.4, 1.4, 2.4, 3.4, 4.4]
    nosd_index = [0.6, 1.6, 2.6, 3.6, 4.6]
    ddqp_index = [0.8, 1.8, 2.8, 3.8, 4.8]
    plt.xlim(0, 5.2)

    label_index = [0.6, 1.6, 2.6, 3.6, 4.6]
    width = 0.16

    plt.bar(bfg_index, [Abilene[2] / Abilene[3], ANS[2] / ANS[3], AboveNet[2] / AboveNet[3], Integra[2] / Integra[3], BICS[2] / BICS[3]], width, color="#0d7263", label="BFG",
            lw=1, edgecolor="black")
    plt.bar(nosd_index, [Abilene[0] / Abilene[3], ANS[0] / ANS[3], AboveNet[0] / AboveNet[3], Integra[0] / Integra[3], BICS[0] / BICS[3]], width, color='#ef8935', label="NOSD",
            lw=1, edgecolor="black")
    plt.bar(ddqp_index, [Abilene[1] / Abilene[3], ANS[1] / ANS[3], AboveNet[1] / AboveNet[3], Integra[1] / Integra[3], BICS[1] / BICS[3]], width, color='#992020', label="DDQP",
            lw=1, edgecolor="black")
    plt.hlines(1.0, 0, 5.2, colors="#132486", lw = 2, linestyles="--")

    plt.xticks(label_index, labels=labels, fontsize=fontsize)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend(fontsize=15)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(hvretio, 5)
    fig.savefig(path_prefix + 'throughput.png', dpi=dpi)
    plt.show()


def plot_service_time_between_algorithms():
    Abilene = [8343.173427926175, 9235.226813703895, 7461.389462078978, 5774.5096932811175]
    ANS = [9464.014045222082, 9903.124539080174, 7727.514574224005, 6204.83386892512]
    BSO = [7070.879378907696, 7686.74459190949, 6510.699674979215, 5647.944096568549]
    AboveNet = [8522.02231237127, 8374.014029134323, 7947.971236652294, 5247.466649645706]
    Integra = [9728.692357181275, 9570.490522587379, 8917.681929965576, 5477.392653721005]
    BICS = [10086.318975777484, 10096.73950753092, 8639.516362580329, 5390.118595563634]

    plt.ylim(0.75, 2)
    higher_BFG = (Abilene[1] / Abilene[2] + ANS[1] / ANS[2] + AboveNet[1] / AboveNet[2]
                  + Integra[1] / Integra[2] + BICS[1] / BICS[2]) / 5 - 1
    higher_RG = (Abilene[1] / Abilene[3] + ANS[1] / ANS[3] + AboveNet[1] / AboveNet[3]
                 + Integra[1] / Integra[3] + BICS[1] / BICS[3]) / 5 - 1
    higher_NOSD = (Abilene[1] / Abilene[0] + ANS[1] / ANS[0] + AboveNet[1] / AboveNet[0]
                   + Integra[1] / Integra[0] + BICS[1] / BICS[0]) / 5 - 1

    print("service_time: avg higher than RG: {}, avg higher than BFG: {}, avg higher than NOSD: {}".format(higher_RG,
                                                                                                           higher_BFG,
                                                                                                           higher_NOSD))

    plt.ylabel('Service Availability/RG', fontsize=fontsize)

    labels = ["Abilene", "ANS", "AboveNet", "Integra", "BICS"]

    bfg_index = [0.4, 1.4, 2.4, 3.4, 4.4]
    nosd_index = [0.6, 1.6, 2.6, 3.6, 4.6]
    ddqp_index = [0.8, 1.8, 2.8, 3.8, 4.8]
    plt.xlim(0, 5.2)

    label_index = [0.6, 1.6, 2.6, 3.6, 4.6]
    width = 0.16

    plt.bar(bfg_index, [Abilene[2] / Abilene[3], ANS[2] / ANS[3], AboveNet[2] / AboveNet[3], Integra[2] / Integra[3], BICS[2] / BICS[3]], width, color="#0d7263", label="BFG",
            lw=1, edgecolor="black")
    plt.bar(nosd_index, [Abilene[0] / Abilene[3], ANS[0] / ANS[3], AboveNet[0] / AboveNet[3], Integra[0] / Integra[3], BICS[0] / BICS[3]], width, color='#ef8935', label="NOSD",
            lw=1, edgecolor="black")
    plt.bar(ddqp_index, [Abilene[1] / Abilene[3], ANS[1] / ANS[3], AboveNet[1] / AboveNet[3], Integra[1] / Integra[3], BICS[1] / BICS[3]], width, color='#992020', label="DDQP",
            lw=1, edgecolor="black")
    plt.hlines(1.0, 0, 5.2, colors="#132486", lw = 2, linestyles="--")

    plt.xticks(label_index, labels=labels, fontsize=fontsize)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend(fontsize=15)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(hvretio, 5)
    fig.savefig(path_prefix + 'service.png', dpi=dpi)
    plt.show()


def plot_total_reward_between_algorithms():
    Abilene = [1295, 1429.6, 1196, 952]
    ANS = [1435, 1505.8, 1183, 1039]
    BSO = [1541, 1436.6, 1162, 1054]
    AboveNet = [1404, 1410.8, 1295, 916]
    Integra = [1406, 1399.8, 1316, 827]
    BICS = [1644, 1634.4, 1388, 883]

    plt.ylim(0.75, 2)
    higher_BFG = (Abilene[1] / Abilene[2] + ANS[1] / ANS[2] + AboveNet[1] / AboveNet[2]
                  + Integra[1] / Integra[2] + BICS[1] / BICS[2]) / 5 - 1
    higher_RG = (Abilene[1] / Abilene[3] + ANS[1] / ANS[3] + AboveNet[1] / AboveNet[3]
                 + Integra[1] / Integra[3] + BICS[1] / BICS[3]) / 5 - 1
    higher_NOSD = (Abilene[1] / Abilene[0] + ANS[1] / ANS[0] + AboveNet[1] / AboveNet[0]
                   + Integra[1] / Integra[0] + BICS[1] / BICS[0]) / 5 - 1

    print("total reward: avg higher than RG: {}, avg higher than BFG: {}, avg higher than NOSD: {}".format(higher_RG,
                                                                                                           higher_BFG,
                                                                                                           higher_NOSD))

    plt.ylabel('Total Reward/RG', fontsize=fontsize)

    labels = ["Abilene", "ANS", "AboveNet", "Integra", "BICS"]


    bfg_index = [0.4, 1.4, 2.4, 3.4, 4.4]
    nosd_index = [0.6, 1.6, 2.6, 3.6, 4.6]
    ddqp_index = [0.8, 1.8, 2.8, 3.8, 4.8]
    plt.xlim(0, 5.2)

    label_index = [0.6, 1.6, 2.6, 3.6, 4.6]
    width = 0.16

    plt.bar(bfg_index, [Abilene[2] / Abilene[3], ANS[2] / ANS[3], AboveNet[2] / AboveNet[3], Integra[2] / Integra[3], BICS[2] / BICS[3]], width, color="#0d7263", label="BFG",
            lw=1, edgecolor="black")
    plt.bar(nosd_index, [Abilene[0] / Abilene[3], ANS[0] / ANS[3], AboveNet[0] / AboveNet[3], Integra[0] / Integra[3], BICS[0] / BICS[3]], width, color='#ef8935', label="NOSD",
            lw=1, edgecolor="black")
    plt.bar(ddqp_index, [Abilene[1] / Abilene[3], ANS[1] / ANS[3], AboveNet[1] / AboveNet[3], Integra[1] / Integra[3], BICS[1] / BICS[3]], width, color='#992020', label="DDQP",
            lw=1, edgecolor="black")
    plt.hlines(1.0, 0, 5.2, colors="#132486", lw = 2, linestyles="--")

    plt.xticks(label_index, labels=labels, fontsize=fontsize)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend(fontsize=15)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(hvretio, 5)
    fig.savefig(path_prefix + 'total_reward.png', dpi=dpi)
    plt.show()


def plot_running_time_cost_between_algorithms():
    Abilene = np.array([53.0, 3.0, 52.0, 10.0]) / 3
    ANS = np.array([184.0, 4.0, 176.0, 18.0]) / 3
    BSO = np.array([202.0, 4.0, 193.0, 20.0]) / 3
    AboveNet = np.array([345.0, 4.0, 323.0, 30.0]) / 3
    Integra = np.array([528.0, 5.0, 519.0, 62.0]) / 3
    BICS = np.array([1094.0, 6.0, 1036.0, 101.0]) / 3

    higher_NOSD = (Abilene[0] / Abilene[1] + ANS[0] / ANS[1] + AboveNet[0] / AboveNet[1] + Integra[0] / Integra[1] +
                   BICS[0] / BICS[1]) / 5
    higher_BFG = (Abilene[2] / Abilene[1] + ANS[2] / ANS[1] + AboveNet[2] / AboveNet[1] + Integra[2] / Integra[1] +
                  BICS[2] / BICS[1]) / 5
    higher_RG = (Abilene[3] / Abilene[1] + ANS[3] / ANS[1] + AboveNet[3] / AboveNet[1] + Integra[3] / Integra[1] + BICS[
        3] / BICS[1]) / 5
    save_higher_NOSD = 1 - (
                Abilene[1] / Abilene[0] + ANS[1] / ANS[0] + AboveNet[1] / AboveNet[0] + Integra[1] / Integra[0] + BICS[
            1] / BICS[0]) / 5
    save_higher_BFG = 1 - (
                Abilene[1] / Abilene[2] + ANS[1] / ANS[2] + AboveNet[1] / AboveNet[2] + Integra[1] / Integra[2] + BICS[
            1] / BICS[2]) / 5
    save_higher_RG = 1 - (
                Abilene[1] / Abilene[3] + ANS[1] / ANS[3] + AboveNet[1] / AboveNet[3] + Integra[1] / Integra[3] + BICS[
            1] / BICS[3]) / 5
    print("running_time_cost: avg higher than NOSD: {}, avg higher than RG: {}, avg higher than BFG: {}".format(
        higher_NOSD, higher_RG, higher_BFG))
    print("running_time_cost: avg higher than NOSD: {}, avg higher than RG: {}, avg higher than BFG: {}".format(
        save_higher_NOSD, save_higher_RG, save_higher_BFG))

    plt.ylim(1, 1000)
    plt.ylabel('Running Time Cost (ms)', fontsize=fontsize)
    plt.yscale('log')
    labels = ["Abilene", "ANS", "AboveNet", "Integra", "BICS"]

    rg_index = [0.3, 1.3, 2.3, 3.3, 4.3]
    bfg_index = [0.5, 1.5, 2.5, 3.5, 4.5]
    nosd_index = [0.7, 1.7, 2.7, 3.7, 4.7]
    ddqp_index = [0.9, 1.9, 2.9, 3.9, 4.9]
    plt.xlim(0, 5.2)

    label_index = [0.6, 1.6, 2.6, 3.6, 4.6]
    width = 0.16

    plt.bar(rg_index, [Abilene[3], ANS[3], AboveNet[3], Integra[3], BICS[3]], width, color="#132486", label="RG",
            lw=1, edgecolor="black")
    plt.bar(bfg_index, [Abilene[2], ANS[2], AboveNet[2], Integra[2], BICS[2]], width, color="#0d7263", label="BFG",
            lw=1, edgecolor="black")
    plt.bar(nosd_index, [Abilene[0], ANS[0], AboveNet[0], Integra[0], BICS[0]], width, color='#ef8935', label="NOSD",
            lw=1, edgecolor="black")
    plt.bar(ddqp_index, [Abilene[1], ANS[1], AboveNet[1], Integra[1], BICS[1]], width, color='#992020', label="DDQP",
            lw=1, edgecolor="black")

    plt.xticks(label_index, labels=labels, fontsize=fontsize)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend(fontsize=15)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(hvretio, 5)
    fig.savefig(path_prefix + 'runningtime.png', dpi=dpi)
    plt.show()


def plot_reward_trace():
    window_size = 3
    if pf == "Windows":
        TRACE_FILE = "model\\trace.pkl"
    elif pf == "Linux":
        TRACE_FILE = "model/trace.pkl"
    with open(TRACE_FILE, 'rb') as f:
        reward_trace = pickle.load(f)  # read file and build object
        index_len = len(reward_trace)
        for i in range(index_len):
            if i >= window_size and index_len - i > window_size:
                sum = 0
                for j in range(i - window_size, i + window_size + 1):
                    sum += reward_trace[j]
                sum = sum / (window_size * 2 + 1)
                reward_trace[i] = sum
        reward_trace = reward_trace[0:1230]
        index_len = len(reward_trace)

        nosd_height = 1412
        rg_height = 1064
        bfg_height = 1249
        line_nosd = [(0, nosd_height), (index_len, nosd_height)]
        line_rg = [(0, rg_height), (index_len, rg_height)]
        line_bfg = [(0, bfg_height), (index_len, bfg_height)]
        (line_tgt_xs, line_tgt_ys) = zip(*line_nosd)
        (line_RG_xs, line_RG_ys) = zip(*line_rg)
        (line_BFG_xs, line_BFG_ys) = zip(*line_bfg)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylabel('Avg. Accepted Numbers', fontsize=fontsize)
        plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
        plt.ylim(0, 1800)
        ax.add_line(Line2D(line_tgt_xs, line_tgt_ys, color="#132486", linestyle="--"))
        ax.add_line(Line2D(line_RG_xs, line_RG_ys, color="#ef8935", linestyle="--"))
        ax.add_line(Line2D(line_BFG_xs, line_BFG_ys, color="#0d7263", linestyle="--"))
        ax.annotate('NOSD',
                    xy=(index_len - 90, nosd_height),
                    xytext=(0, -20),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color="#132486", fontsize=fontsize)
        ax.annotate('RG',
                    xy=(index_len - 90, rg_height),
                    xytext=(0, -20),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color="#ef8935", fontsize=fontsize)
        ax.annotate('BFG',
                    xy=(index_len - 90, bfg_height),
                    xytext=(0, -20),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color="#0d7263", fontsize=fontsize)

        ax.arrow(19, rg_height, 200, -400,
                 length_includes_head=True,  # 增加的长度包含箭头部分
                 head_width=25, head_length=25, fc='#ef8935', ec='#ef8935')
        ax.annotate('19th',
                    xy=(220, 650),
                    xytext=(0, -20),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color="#ef8935", fontsize=fontsize)

        ax.arrow(121, bfg_height, 400, -500,
                 length_includes_head=True,  # 增加的长度包含箭头部分
                 head_width=25, head_length=25, fc='#0d7263', ec='#0d7263')
        ax.annotate('121th',
                    xy=(530, 750),
                    xytext=(0, -20),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color="#0d7263", fontsize=fontsize)

        ax.arrow(676, nosd_height, 200, -500,
                 length_includes_head=True,  # 增加的长度包含箭头部分
                 head_width=25, head_length=25, fc='#132486', ec='#132486')
        ax.annotate('676th',
                    xy=(880, 920),
                    xytext=(0, -20),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color="#132486", fontsize=fontsize)

        ax.set_xlabel("Episode", fontsize=fontsize)
        plt.plot(reward_trace, color="#1b77aa", linewidth=3, label="DDQP Target Agent")
        plt.legend(fontsize=fontsize)
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(hvretio, 5)
        fig.savefig(path_prefix + 'training.png', dpi=dpi)
        plt.show()


def plot_acceptance_rate_between_configs():
    #
    Abilene = [0.616458618782074, 0.5339940272295755, 0.5166870551249897, 0.4778743878729014, 0.3362014460219059]
    ANS = [0.6892168906377846, 0.5859496133069216, 0.560774897269359, 0.49892813635278854, 0.38778679716881087]
    AboveNet = [0.6516257891948044, 0.5376352966567849, 0.515050139252096, 0.4622908573702517, 0.38002809808679416]
    BSO = [0.699429599640426, 0.5727141004096132, 0.5450564672547901, 0.48927261955906465, 0.3816132537780993]
    Integra = [0.5973263582537216, 0.5147366169821552, 0.48677529863686847, 0.4732296532951182, 0.34189013654465067]
    BICS = [0.6709822907091505, 0.6028478648800558, 0.5888755575310769, 0.5426460956242856, 0.46330987488366426]
    plt.ylabel('Acceptance Rate/NoBackup', fontsize=fontsize)

    AboveNet = np.array(AboveNet) / AboveNet[0]

    plt.ylim(0, 1.2)
    plt.xlim(0, 5.2)
    labels = ["NoBackup", "Aggressive", "Normal", "MaxR", "FullyR"]
    index = [0.6, 1.6, 2.6, 3.6, 4.6]
    width = 0.4
    plt.bar(index, AboveNet, width, color='#ffffff', lw=1, edgecolor="#132486", hatch="//")
    plt.plot(index, AboveNet, color='#132486', lw=1, marker="^")
    plt.xticks(index, labels=labels, fontsize=fontsize)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(hvretio, 5)
    fig.savefig(path_prefix + 'acceptance_c.png', dpi=dpi)
    plt.show()


def plot_throughput_between_configs():
    Abilene = [141972.6, 120997.6, 113001.8, 103279.2, 75098.8]
    ANS = [156911.6, 132246.6, 125832.2, 112276.6, 84698.4]
    AboveNet = [153112.2, 124569.8, 118580.4, 106740.6, 85957.8]
    Integra = [137573.2, 117477.6, 112372.0, 106579.0, 78192.0]
    BICS = [161132.2, 140866.8, 136108.6, 125250.6, 107186.2]
    plt.ylabel('Throughput/NoBackup', fontsize=fontsize)

    AboveNet = np.array(AboveNet) / AboveNet[0]

    plt.ylim(0, 1.25)
    plt.xlim(0, 5.2)
    labels = ["NoBackup", "Aggressive", "Normal", "MaxR", "FullyR"]
    index = [0.6, 1.6, 2.6, 3.6, 4.6]
    width = 0.4
    plt.bar(index, AboveNet, width, color='#ffffff', lw=1, edgecolor="#992020", hatch="//")
    plt.plot(index, AboveNet, color='#992020', lw=1, marker="^")
    plt.xticks(index, labels=labels, fontsize=fontsize)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(hvretio, 5)
    fig.savefig(path_prefix + 'throughput_c.png', dpi=dpi)
    plt.show()


def plot_service_time_between_configs():
    Abilene = [8487.599674065888, 10139.339467384647, 9915.841114782414, 8918.123742070178, 6658.523629180175]
    ANS = [9095.190373651156, 10986.839758664599, 10846.192826595234, 9972.638182515682, 7611.680673023506]
    BSO = [6813.933177247144, 8537.23304260184, 8312.284219118028, 7633.906282503199, 6278.393114141079]
    AboveNet = [6835.440256333747, 9407.31365279395, 9204.597169421892, 8445.231275786758, 6836.6715336073785]
    Integra = [8489.987265138121, 10096.104548752031, 9913.459418297749, 9333.334116695238, 6966.20434854566]
    BICS = [7587.377662160613, 10661.351412752329, 10666.307790843455, 9898.793860722668, 8775.978207528173]
    plt.ylabel('Service Availability/NoBackup', fontsize=fontsize)

    AboveNet = np.array(AboveNet) / AboveNet[0]

    plt.ylim(0, 1.6)
    plt.xlim(0, 5.2)
    labels = ["NoBackup", "Aggressive", "Normal", "MaxR", "FullyR"]
    index = [0.6, 1.6, 2.6, 3.6, 4.6]
    width = 0.4
    plt.bar(index, AboveNet, width, color='#ffffff', lw=1, edgecolor="#0d7263", hatch="//")
    plt.plot(index, AboveNet, color='#0d7263', lw=1, marker="^")
    plt.xticks(index, labels=labels, fontsize=fontsize)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(hvretio, 5)
    fig.savefig(path_prefix + 'service_time_c.png', dpi=dpi)
    plt.show()


def plot_total_reward_between_configs():
    Abilene = [1863.2, 1615.2, 1523.4, 1382.6, 1010.8]
    ANS = [2043.0, 1745.8, 1674.0, 1507.8, 1138.2]
    AboveNet = [1975.2, 1636.2, 1567.6, 1416.8, 1134.0]
    Integra = [1764.6, 1527.8, 1462.6, 1394.8, 1025.4]
    BICS = [2039.4, 1797.4, 1751.6, 1617.8, 1394.6]
    plt.ylabel('Total Reward/NoBackup', fontsize=fontsize)

    AboveNet = np.array(AboveNet) / AboveNet[0]

    plt.ylim(0, 1.25)
    plt.xlim(0, 5.2)
    labels = ["NoBackup", "Aggressive", "Normal", "MaxR", "FullyR"]
    index = [0.6, 1.6, 2.6, 3.6, 4.6]
    width = 0.4
    plt.bar(index, AboveNet, width, color='#ffffff', lw=1, edgecolor="#ef8935", hatch="//")
    plt.plot(index, AboveNet, color='#ef8935', lw=1, marker="^")
    plt.xticks(index, labels=labels, fontsize=fontsize)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(hvretio, 5)
    fig.savefig(path_prefix + 'total_reward_c.png', dpi=dpi)
    plt.show()


def plot_real_fail_rate_between_configs():
    Abilene = [1.0, 0.180760440473717, 0.1039537983118614, 0.046005951018539715, 0]
    ANS = [1.0, 0.15859766277128548, 0.09221579242113123, 0.0325489333626567, 0]
    BSO = [1.0, 0.20898550724637682, 0.12234595397178916, 0.0576955424726661, 0]
    AboveNet = [1.0, 0.09055182577370231, 0.06274443121918041, 0.01736881005173688, 0]
    fault_tolerance = np.ones(5) - [1.0, 0.09055182577370231, 0.06274443121918041, 0.01736881005173688, 0]
    Integra = [1.0, 0.11355735805330243, 0.07495164410058026, 0.03045557513214196, 0]
    BICS = [1.0, 0.10935887988209285, 0.05439076268611364, 0.026345933562428408, 0]
    plt.ylabel('Fault Tolerance', fontsize=fontsize)

    plt.ylim(0, 1.1)
    plt.xlim(0, 5.2)
    labels = ["NoBackup", "Aggressive", "Normal", "MaxR", "FullyR"]
    index = [0.6, 1.6, 2.6, 3.6, 4.6]
    width = 0.4
    plt.bar(index, fault_tolerance, width, color='#ffffff', lw=1, edgecolor="#000000", hatch="//")
    plt.plot(index, fault_tolerance, color='#000000', lw=1, marker="^")
    plt.xticks(index, labels=labels, fontsize=fontsize)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(hvretio, 5)
    fig.savefig(path_prefix + 'fail_rate_c.png', dpi=dpi)
    plt.show()


if __name__ == "__main__":
    fig = plt.figure()
    plot_acceptance_rate_between_algorithms()
    plot_throughput_between_algorithms()
    plot_service_time_between_algorithms()
    plot_total_reward_between_algorithms()
    plot_running_time_cost_between_algorithms()
    # plot_reward_trace()
    plot_acceptance_rate_between_configs()
    plot_real_fail_rate_between_configs()
    plot_service_time_between_configs()
    plot_total_reward_between_configs()
    plot_throughput_between_configs()
