import pickle
from typing import List

def compare(reward_trace: List, target: int, window_size: int):
    """
    calculate which time the training agent reach target value
    :param reward_trace: the list of reward trace
    :param target: target value
    :param window_size: window size
    :return: index
    """
    avg = []
    origin_len = len(reward_trace)
    for i in range(len(reward_trace)):
        if i < window_size or  i + window_size >= origin_len:
            avg.append(reward_trace[i])
        else:
            sum = 0
            for j in range(window_size):
                sum += reward_trace[i + j + 1] + reward_trace[i - j - 1]
            sum += reward_trace[i]
            avg.append(sum / (window_size * 2 + 1))
    for i in range(len(avg)):
        if avg[i] >= target:
            return i
    return -1



if __name__ == "__main__":
    with open("model\\trace.pkl", 'rb') as f:
        reward_trace = pickle.load(f)  # read file and build object

    # print(compare(reward_trace, 924, 5))
    # reward_trace = reward_trace[0:1200]
    # for i in range(len(reward_trace)):
    #     if reward_trace[i] > 100:
    #         reward_trace[i] -= 100

    print(compare(reward_trace, 1064, 3))
    print(compare(reward_trace, 1249, 3))
    print(compare(reward_trace, 1412, 3))
    # with open("model\\trace.pkl", 'wb') as f:  # open file with write-mode
    #     pickle.dump(reward_trace, f)  # serialize and save object

