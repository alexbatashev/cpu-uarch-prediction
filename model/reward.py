import torch.nn as nn
import torch


def reward_function(port_pressures, predicted_cycles, true_cycles, nodes):
    criterion = nn.SmoothL1Loss()
    base_reward = criterion(torch.tensor([predicted_cycles]), torch.tensor([true_cycles]))

    constraint_violation_penalty = 0

    ports_num = len(port_pressures[0])
    for p in port_pressures:
        if len(p) != ports_num:
            constraint_violation_penalty = 100
            return base_reward - constraint_violation_penalty
        total = sum(p)
        if total < 1.0:
            constraint_violation_penalty += 10
        # It is unrealistic that a micro-operation occupies
        # port for less than 1/3 of the CPU cycle
        for n in p:
            if n < 0.3:
                constraint_violation_penalty += 3


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

    return base_reward * constraint_violation_penalty
