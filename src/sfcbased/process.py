from sfcbased.utils import *


def deploy_sfc_item(model: Model, sfc_index: int, decision_maker: DecisionMaker, time: int, state: List, test_env: TestEnv):
    """
    deploy sfc i in time t based on current state

    we needn't to make sure the decision can be placed because we can get flag from subroutine

    :param model: model
    :param sfc_index: current sfc index
    :param decision_maker: make decision
    :param time: time
    :param state: state
    :param test_env: test environment
    :return: Decision
    """
    assert model.sfc_list[sfc_index].state == State.Undeployed
    flag, decision = make_decision(model, decision_maker, sfc_index, state, test_env)

    # Undeployed→Failed
    if not flag:
        model.sfc_list[sfc_index].set_state(time, sfc_index, State.Failed)
        return decision

    # backup condition
    if test_env != TestEnv.NoBackup:
        model.sfc_list[sfc_index].standby_sfc.server = decision.standby_server
        model.sfc_list[sfc_index].standby_sfc.path_s2c = decision.standby_path_s2c
        model.sfc_list[sfc_index].standby_sfc.path_c2d = decision.standby_path_c2d
        model.sfc_list[sfc_index].update_path = decision.update_path
        deploy_standby(model, sfc_index, test_env)

    # Undeployed→Normal
    model.sfc_list[sfc_index].active_sfc.server = decision.active_server
    model.sfc_list[sfc_index].active_sfc.path_s2c = decision.active_path_s2c
    model.sfc_list[sfc_index].active_sfc.path_c2d = decision.active_path_c2d
    deploy_active(model, sfc_index, test_env)
    model.sfc_list[sfc_index].set_state(time, sfc_index, State.Normal)
    return decision


def deploy_sfcs_in_timeslot(model: Model, decision_maker: DecisionMaker, time: int, state: List, test_env: TestEnv):
    """
    deploy the sfcs located in given time slot t

    this method is mainly used by classic method, for they don't have to make decision based on the
    state of the environment used by reinforcement learning approach

    :param model: model
    :param decision_maker: make decision
    :param time: time
    :param state: state
    :param test_env: test environment
    :return: None
    """
    for i in range(len(model.sfc_list)):
        if time <= model.sfc_list[i].time < time + 1:
            deploy_sfc_item(model, i, decision_maker, time, state, test_env)


def deploy_active(model: Model, sfc_index: int, test_env: TestEnv):
    """
    start active instance of given sfc

    it must can be deployed because we have examined it, so we don't need to examine again

    :param model: model
    :param sfc_index: sfc index
    :param test_env: test environment
    :return: None
    """

    # occupy computing resource
    model.topo.nodes[model.sfc_list[sfc_index].active_sfc.server]["active"] += model.sfc_list[
        sfc_index].computing_resource

    # occupy path bandwidth
    path_s2c = model.sfc_list[sfc_index].active_sfc.path_s2c
    path_c2d = model.sfc_list[sfc_index].active_sfc.path_c2d
    for i in range(len(path_s2c) - 1):
        model.topo.edges[path_s2c[i], path_s2c[i + 1]]["active"] += model.sfc_list[sfc_index].tp
    for i in range(len(path_c2d) - 1):
        model.topo.edges[path_c2d[i], path_c2d[i + 1]]["active"] += model.sfc_list[sfc_index].tp

    # update path
    if test_env != TestEnv.NoBackup:
        # occupy update bandwidth
        path = model.sfc_list[sfc_index].update_path
        for i in range(len(path) - 1):
            model.topo.edges[path[i], path[i + 1]]["active"] += model.sfc_list[sfc_index].update_tp


def deploy_standby(model: Model, sfc_index: int, test_env: TestEnv):
    """
    deploy standby instance of given sfc

    it must can be deployed because we have examined it, so we don't need to examine again,
    we also don't need to consider update path, because it has been considered when deploying active instance

    :param model: model
    :param sfc_index: sfc index
    :param test_env: test environment
    :return: None
    """
    assert test_env != TestEnv.NoBackup
    sfc = model.sfc_list[sfc_index]
    standby_server = model.topo.nodes[sfc.standby_sfc.server]
    path_s2c = sfc.standby_sfc.path_s2c
    path_c2d = sfc.standby_sfc.path_c2d

    # MaxReservation
    if test_env == TestEnv.MaxReservation:
        # computing resource reservation
        standby_server["sbsfcs"].add(sfc_index)
        if sfc.computing_resource > standby_server["reserved"]:
            standby_server["reserved"] = sfc.computing_resource
            standby_server["max_sbsfc_index"] = sfc_index

        # bandwidth reservation
        for i in range(len(path_s2c) - 1):
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["sbsfcs_s2c"].add(sfc_index)
            if sfc.tp > model.topo.edges[path_s2c[i], path_s2c[i + 1]]["reserved"]:
                model.topo.edges[path_s2c[i], path_s2c[i + 1]]["reserved"] = sfc.tp
                model.topo.edges[path_s2c[i], path_s2c[i + 1]]["max_sbsfc_index"] = sfc_index
        for i in range(len(path_c2d) - 1):
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["sbsfcs_c2d"].add(sfc_index)
            if sfc.tp > model.topo.edges[path_c2d[i], path_c2d[i + 1]]["reserved"]:
                model.topo.edges[path_c2d[i], path_c2d[i + 1]]["reserved"] = sfc.tp
                model.topo.edges[path_c2d[i], path_c2d[i + 1]]["max_sbsfc_index"] = sfc_index

    # FullyReservation
    if test_env == TestEnv.FullyReservation:
        # computing resource reservation
        standby_server["sbsfcs"].add(sfc_index)
        standby_server["reserved"] += sfc.computing_resource

        # bandwidth reservation
        for i in range(len(path_s2c) - 1):
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["sbsfcs_s2c"].add(sfc_index)
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["reserved"] += sfc.tp
        for i in range(len(path_c2d) - 1):
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["sbsfcs_c2d"].add(sfc_index)
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["reserved"] += sfc.tp


def active_failed(model: Model, sfc_index: int, test_env: TestEnv):
    """
    handle the active instance failed condition, including resource reclaiming

    mainly three parts:
    1. computing resource occupied by active instance
    2. flow bandwidth occupied by active instance
    3. update bandwidth occupied by active instance

    :param model: model
    :param sfc_index: the index of sfc
    :param test_env: test environment
    :return: None
    """
    sfc = model.sfc_list[sfc_index]

    # release computing resource
    model.topo.nodes[sfc.active_sfc.server]["active"] -= sfc.computing_resource

    # release path bandwidth
    path_s2c = sfc.active_sfc.path_s2c
    path_c2d = sfc.active_sfc.path_c2d
    for i in range(len(path_s2c) - 1):
        model.topo.edges[path_s2c[i], path_s2c[i + 1]]["active"] -= sfc.tp
    for i in range(len(path_c2d) - 1):
        model.topo.edges[path_c2d[i], path_c2d[i + 1]]["active"] -= sfc.tp

    # release update path bandwidth
    if test_env != TestEnv.NoBackup:
        path = sfc.update_path
        for i in range(len(path) - 1):
            model.topo.edges[path[i], path[i + 1]]["active"] -= sfc.update_tp


def remove_reservation_for_MaxReservation(model: Model, sfc_index: int):
    """
    remove reservation of given sfc under MaxReservation

    mainly used in two condition:
    1. when standby start failed, we need to remove the reservation
    2. when standby start successfully, we need to remove the reservation
    3. when standby doesn't start and expired, we need to remove the reservation
    we don't need to remove the reservation under condition:
    1. NoBackup: for we didn't backup
    2. Aggressive: for we didn't record its reservation in the property of sfc server instance
    3. Normal: as same as Aggressive

    :param model: model
    :param sfc_index: sfc index
    :return: None
    """
    # remove computing resource reservation
    model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["sbsfcs"].remove(sfc_index)
    if model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["max_sbsfc_index"] == sfc_index:
        max_value = 0
        max_index = -1
        for index in model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["sbsfcs"]:
            if model.sfc_list[index].computing_resource > max_value:
                max_value = model.sfc_list[index].computing_resource
                max_index = index
        model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["max_sbsfc_index"] = max_index
        model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["reserved"] = max_value

    # remove bandwidth reservation
    path_s2c = model.sfc_list[sfc_index].standby_sfc.path_s2c
    path_c2d = model.sfc_list[sfc_index].standby_sfc.path_c2d
    for i in range(len(path_s2c) - 1):
        model.topo.edges[path_s2c[i], path_s2c[i + 1]]["sbsfcs_s2c"].remove(sfc_index)
        if model.topo.edges[path_s2c[i], path_s2c[i + 1]]["max_sbsfc_index"] == sfc_index:
            max_value = 0
            max_index = -1
            for index in model.topo.edges[path_s2c[i], path_s2c[i + 1]]["sbsfcs_s2c"]:
                if model.sfc_list[index].tp > max_value:
                    max_value = model.sfc_list[index].tp
                    max_index = index
            for index in model.topo.edges[path_s2c[i], path_s2c[i + 1]]["sbsfcs_c2d"]:
                if model.sfc_list[index].tp > max_value:
                    max_value = model.sfc_list[index].tp
                    max_index = index
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["max_sbsfc_index"] = max_index
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["reserved"] = max_value
    for i in range(len(path_c2d) - 1):
        model.topo.edges[path_c2d[i], path_c2d[i + 1]]["sbsfcs_c2d"].remove(sfc_index)
        if model.topo.edges[path_c2d[i], path_c2d[i + 1]]["max_sbsfc_index"] == sfc_index:
            max_value = 0
            max_index = -1
            for index in model.topo.edges[path_c2d[i], path_c2d[i + 1]]["sbsfcs_s2c"]:
                if model.sfc_list[index].tp > max_value:
                    max_value = model.sfc_list[index].tp
                    max_index = index
            for index in model.topo.edges[path_c2d[i], path_c2d[i + 1]]["sbsfcs_c2d"]:
                if model.sfc_list[index].tp > max_value:
                    max_value = model.sfc_list[index].tp
                    max_index = index
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["max_sbsfc_index"] = max_index
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["reserved"] = max_value


def standby_start(model: Model, sfc_index: int, test_env: TestEnv):
    """
    handle the standby instance start condition

    :param model: model
    :param sfc_index: sfc index
    :param test_env: test environment
    :return: bool
    """
    assert test_env != TestEnv.NoBackup

    sfc = model.sfc_list[sfc_index]
    standby_server = model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]
    path_s2c = sfc.standby_sfc.path_s2c
    path_c2d = sfc.standby_sfc.path_c2d

    # FullyReservation, is this condition, at any time the standby instance can start and we don't need to examine
    if test_env == TestEnv.FullyReservation:
        standby_server["active"] += sfc.computing_resource
        standby_server["reserved"] -= sfc.computing_resource
        standby_server["sbsfcs"].remove(sfc_index)
        for i in range(len(path_s2c) - 1):
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["sbsfcs_s2c"].remove(sfc_index)
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["active"] += sfc.tp
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["reserved"] -= sfc.tp
        for i in range(len(path_c2d) - 1):
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["sbsfcs_c2d"].remove(sfc_index)
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["active"] += sfc.tp
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["reserved"] -= sfc.tp
        return True

    # others(Aggressive, Normal, MaxReservation)
    # examination(if is MaxReservation and start failed, then remove the reservation)
    failed = False
    remaining = standby_server["computing_resource"] - standby_server["active"]
    if remaining < sfc.computing_resource:
        failed = True
    if not failed:
        for i in range(len(path_s2c) - 1):
            remaining = model.topo.edges[path_s2c[i], path_s2c[i + 1]]["bandwidth"] - \
                        model.topo.edges[path_s2c[i], path_s2c[i + 1]]["active"]
            if remaining < sfc.tp:
                failed = True
                break
    if not failed:
        for i in range(len(path_c2d) - 1):
            remaining = model.topo.edges[path_c2d[i], path_c2d[i + 1]]["bandwidth"] - \
                        model.topo.edges[path_c2d[i], path_c2d[i + 1]]["active"]
            if remaining < sfc.tp:
                failed = True
                break

    # MaxReservation - whether success or fail, we both need to remove reservation for Max
    if test_env == TestEnv.MaxReservation:
        remove_reservation_for_MaxReservation(model, sfc_index)

    # failed and remove reservation (MaxReservation)
    if failed:
        return False

    # start success
    standby_server["active"] += sfc.computing_resource
    for i in range(len(path_s2c) - 1):
        model.topo.edges[path_s2c[i], path_s2c[i + 1]]["active"] += sfc.tp
    for i in range(len(path_c2d) - 1):
        model.topo.edges[path_c2d[i], path_c2d[i + 1]]["active"] += sfc.tp
    return True


def standby_failed(model: Model, sfc_index: int, test_env: TestEnv):
    """
    handle the standby instance failed condition, including resource reclaiming

    :param model: model
    :param sfc_index: sfc index
    :param test_env: test environment
    :return: None
    """
    assert test_env != TestEnv.NoBackup

    # release computing resource
    model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["active"] -= model.sfc_list[
        sfc_index].computing_resource

    # release path bandwidth
    path_s2c = model.sfc_list[sfc_index].standby_sfc.path_s2c
    path_c2d = model.sfc_list[sfc_index].standby_sfc.path_c2d
    for i in range(len(path_s2c) - 1):
        model.topo.edges[path_s2c[i], path_s2c[i + 1]]["active"] -= model.sfc_list[sfc_index].tp
    for i in range(len(path_c2d) - 1):
        model.topo.edges[path_c2d[i], path_c2d[i + 1]]["active"] -= model.sfc_list[sfc_index].tp


def remove_expired_active(model: Model, sfc_index: int, test_env: TestEnv):
    """
    remove active when time expired

    :param model: model
    :param sfc_index: sfc index
    :param test_env: test environment
    :return: None
    """

    # we don't need to handle the backup state for the original active instance is not running
    if model.sfc_list[sfc_index].state == State.Normal:
        active_failed(model, sfc_index, test_env)


def remove_expired_standby(model: Model, sfc_index: int, test_env: TestEnv):
    """
    remove standby for time expired

    consider reclaim resource for running standby instance,
    consider remove reservation when active not ever failed

    :param model: model
    :param sfc_index: sfc index
    :param test_env: test environment
    :return: None
    """
    assert test_env != TestEnv.NoBackup

    # is running
    if model.sfc_list[sfc_index].state == State.Backup:
        standby_failed(model, sfc_index, test_env)
        return None

    # not running
    assert model.sfc_list[sfc_index].state == State.Normal

    # FullyReservation - remove reservation
    if test_env == TestEnv.FullyReservation:
        model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["sbsfcs"].remove(sfc_index)
        model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["reserved"] -= model.sfc_list[
            sfc_index].computing_resource
        path_s2c = model.sfc_list[sfc_index].standby_sfc.path_s2c
        path_c2d = model.sfc_list[sfc_index].standby_sfc.path_c2d
        for i in range(len(path_s2c) - 1):
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["sbsfcs_s2c"].remove(sfc_index)
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["reserved"] -= model.sfc_list[sfc_index].tp
        for i in range(len(path_c2d) - 1):
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["sbsfcs_c2d"].remove(sfc_index)
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["reserved"] -= model.sfc_list[sfc_index].tp

    # MaxReservation - remove reservation
    elif test_env == TestEnv.MaxReservation:
        remove_reservation_for_MaxReservation(model, sfc_index)


def state_transition_and_resource_reclaim(model: Model, time: int, test_env: TestEnv, failed_instances: List[Instance]):
    """
    in each time slot, handle state transition and reclaim resources

    :param model: model
    :param time: time slot
    :param test_env: test environment
    :param failed_instances: failed instances
    :return: None
    """
    # random failed condition
    for ins in failed_instances:
        index = ins.sfc_index
        is_active = ins.is_active

        # Normal→Backup and Normal→Broken
        if model.sfc_list[index].state == State.Normal:
            assert is_active is True
            active_failed(model, index, test_env)
            if test_env == TestEnv.NoBackup: # NoBackup don't need to start standby instance
                model.sfc_list[index].set_state(time, index, State.Broken, BrokenReason.ActiveDamage)
                continue
            if standby_start(model, index, test_env):
                model.sfc_list[index].set_state(time, index, State.Backup)
            else:
                model.sfc_list[index].set_state(time, index, State.Broken, BrokenReason.StandbyStartFailed)

        # Backup→Broken
        elif model.sfc_list[index].state == State.Backup:
            assert is_active is False
            standby_failed(model, index, test_env)
            model.sfc_list[index].set_state(time, index, State.Broken, BrokenReason.StandbyDamage)

    # time expired condition
    for index in range(len(model.sfc_list)):
        if (model.sfc_list[index].state == State.Normal or model.sfc_list[index].state == State.Backup) and model.sfc_list[index].time + model.sfc_list[
            index].TTL < time:
            if model.sfc_list[index].state == State.Normal:
                remove_expired_active(model, index, test_env)
            if test_env != TestEnv.NoBackup:
                remove_expired_standby(model, index, test_env)
            model.sfc_list[index].set_state(time, index, State.Broken, BrokenReason.TimeExpired)


def process_time_slot(model: Model, decision_maker: DecisionMaker, time: int, test_env: TestEnv,
                      state: List, failed_instances: List[Instance]):
    """
    function used to simulate within given time slot

    :param model: model
    :param decision_maker: decision maker
    :param time: time
    :param test_env: test environment
    :param state: state
    :param failed_instances: failed instances
    :return: None
    """

    # First, handle the state transition and resources reclaim
    # The transition:
    # 1. - Normal→Backup;
    # 2. - Normal→Broken;
    # 3. - Backup→Broken.
    # are processed in this function
    state_transition_and_resource_reclaim(model, time, test_env, failed_instances)

    # Deploy sfc in this time slot
    # The transition:
    # 1. - Undeployed→Failed;
    # 2. - Undeployed→Normal.
    # are processed in this function
    deploy_sfcs_in_timeslot(model, decision_maker, time, state, test_env)
