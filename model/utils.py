import io
import math
import os
import json
import torch
import yaml
from torch_geometric.data import Data, Dataset
from torch import nn
from torch_geometric.utils import add_self_loops
import model.mc_dataset_pb2
import numpy as np
import matplotlib.pyplot as plt


class BasicBlockDataset(Dataset):
    def __init__(self, dataset_path, embedding_size=0, max_cache_miss=0, min_nodes=3, min_cycles=1., max_context_switches=0, dtype=torch.float32):
        super().__init__(None, None, None)
        with open(dataset_path, 'rb') as f:
            dataset = model.mc_dataset_pb2.MCDataset()
            dataset.ParseFromString(f.read())

            #self.num_opcodes = dataset.data[0].graph.num_opcodes
            self.num_opcodes = 16
            self.has_virtual_root = dataset.data[0].graph.has_virtual_root

            self.embeddings = []
            self.raw = []
            self.measurements = []
            for d in dataset.data:
                if getattr(d.metrics, 'measured_num_runs', 0) == 0:
                    continue
                if d.metrics.measured_cycles < 0:
                    continue
                if d.metrics.measured_cycles / getattr(d.metrics, 'measured_num_runs') > 50:
                    continue
                if d.metrics.workload_cache_misses - d.metrics.noise_cache_misses != 0:
                    continue
                if d.metrics.workload_context_switches - d.metrics.noise_context_switches != 0:
                    continue
                # if d.metrics.measured_cycles / 1000 < min_cycles:
                #     continue
                # if d.metrics.total_cache_misses > max_cache_miss:
                #     continue
                # if d.metrics.total_context_switches > max_context_switches:
                #     continue
                # if len(d.graph.nodes) < min_nodes:
                #     continue
                nodes = []
                for n in d.graph.nodes:
                    if n.is_virtual_root:
                        nodes.append(torch.zeros(16, dtype=dtype))
                    else:
                        binstr = "{0:b}".format(n.opcode)
                        binary = []
                        for c in reversed(binstr):
                            binary.append(float(int(c)))
                        binary.extend([0] * (16 - len(binstr)))
                        nodes.append(torch.tensor(binary))
                    # ohtype = dtype
                    # if embedding_size > 0:
                    #     ohtype = torch.int
                    # one_hot = torch.zeros([d.graph.num_opcodes], dtype=ohtype)
                    # if not n.is_virtual_root:
                    #     one_hot[n.onehot] = 1
                    # if embedding_size == 0:
                    #     nodes.append(one_hot)
                    # else:
                    #     nodes.append(embed(one_hot))
                nodes = torch.stack(nodes)

                edges = []
                edge_attrs = []
                for e in d.graph.edges:
                    edge_from = getattr(e, 'from', 0)
                    edge_to = getattr(e, 'to', 0)
                    edges.append(torch.tensor([edge_from, edge_to]))
                    edges.append(torch.tensor([edge_to, edge_from]))
                    edge_attrs.append(torch.tensor(1 if getattr(e, 'is_data', False) else 0))
                    edge_attrs.append(torch.tensor(1 if getattr(e, 'is_data', False) else 0))
                if len(edges) == 0:
                    continue
                edges = torch.stack(edges).t().contiguous()
                edge_attrs = torch.stack(edge_attrs)

                self.embeddings.append(Data(x=nodes, edge_index=edges, edge_attr=edge_attrs, y=d.metrics.measured_cycles / getattr(d.metrics, 'measured_num_runs', 1000)))

                #self.measurements.append(d.metrics.measured_cycles / getattr(d.metrics, 'measured_num_runs', 1000))

                raw = {"source": str(d.graph.source),
                       #"edges": d.graph.edges,
                       #"nodes": d.graph.nodes
                       }
                self.raw.append(raw)

    def len(self):
        return len(self.embeddings)

    def get(self, index):
        x = self.embeddings[index]
        # y = self.measurements[index]
        z = self.raw[index]

        return x, z


def unique_rows(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np, axis=0)
    return torch.tensor(unique_np, device=tensor.device, dtype=tensor.dtype)


def get_all_nodes(dataset_path, dtype=torch.float32):
    with open(dataset_path, 'rb') as f:
        dataset = model.mc_dataset_pb2.MCDataset()
        dataset.ParseFromString(f.read())

        num_opcodes = dataset.data[0].graph.num_opcodes

        nodes = []

        for i in range(num_opcodes):
            onehot = torch.zeros(num_opcodes, dtype=dtype)
            onehot[i] = 1
            nodes.append(onehot)

        return torch.stack(nodes)

def save_checkpoint(epoch, model, optimizer, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]


def estimate_cycles(port_pressure_sequence):
    total_cycles = 0

    for instruction in port_pressure_sequence:
        max_pressure = max(instruction)
        total_cycles += max_pressure

    return total_cycles


def print_port_pressure_table(predicted_port_pressures, instructions):
    num_ports = predicted_port_pressures.shape[1]
    header = "Port  |"
    separator = "------"

    for i in range(num_ports):
        header += " {:^7}|".format(i)
        separator += "--------"

    print(header)
    print(separator)

    instr_list = instructions.strip().split('\n')

    for i, row in enumerate(zip(predicted_port_pressures, instr_list)):
        # TODO put useful info here
        row_string = "{:^7}|".format("")
        for pressure in row[0]:
            row_string += " {:<7.2f}|".format(pressure)
        row_string += " {}".format(row[1].strip())
        print(row_string)


def plot_hist(measurements):
    plt.figure()
    plt.plot(60, 60)
    plt.title("Prediction distribution")
    max_val = max(np.max(measurements[0]), np.max(measurements[1]))
    max_val = math.ceil(max_val)
    plt.hist2d(measurements[0], measurements[1], bins=(np.arange(0, max_val, 3), np.arange(0, max_val, 3)))
    plt.xlabel("Measured")
    plt.ylabel("Predicted")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf
