def reward_function(predicted_cycles, true_cycles, port_pressures, nodes):
    base_reward = -abs(predicted_cycles - true_cycles)

    constraint_violation_penalty = 0

    ports_num = len(port_pressures[0])
    for p in port_pressures:
        if len(p) != ports_num:
            constraint_violation_penalty = 100
            return base_reward - constraint_violation_penalty

    # TODO proper mapping
    # 2 -> is_load
    # 3 -> is_store
    # 1 -> is_compute

    pure_memory_instructions = [
        i for i, node in enumerate(nodes) if (node[2] or node[3]) and not node[1]
    ]

    pure_compute_instructions = [
        i for i, node in enumerate(nodes) if not (node[2] or node[3]) and node[1]
    ]

    for i in pure_memory_instructions:
        for j in pure_compute_instructions:
            for k in range(0, ports_num):
                if port_pressures[i][k] != 0 and port_pressures[j][k] != 0:
                    constraint_violation_penalty += 1

    return base_reward - constraint_violation_penalty
