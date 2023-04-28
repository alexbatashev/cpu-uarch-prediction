import torch.nn as nn
import torch


def reward_lsu_compute_separation(port_pressures, nodes, batch_id):
    reward = 0.0
    # TODO proper mapping
    # 2 -> is_load
    # 3 -> is_store
    # 1 -> is_compute

    pure_memory_instructions = [
        i for i, node in enumerate(nodes) if (node["is_load"][batch_id] or node["is_store"][batch_id]) and not node["is_compute"][batch_id]
    ]

    pure_compute_instructions = [
        i for i, node in enumerate(nodes) if not (node["is_load"][batch_id] or node["is_store"][batch_id]) and node["is_compute"][batch_id]
    ]

    for i in pure_memory_instructions:
        for j in pure_compute_instructions:
            for k in range(0, len(port_pressures[0])):
                if port_pressures[i][k] != 0 and port_pressures[j][k] != 0:
                    reward += 0.1

    return reward
