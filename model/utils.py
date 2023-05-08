import os
import json
import torch
import yaml
from torch_geometric.data import Data, Dataset
from torch import nn
from torch_geometric.utils import add_self_loops
import model.mc_dataset_pb2


class BasicBlockDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__(None, None, None)
        with open(dataset_path, 'rb') as f:
            dataset = model.mc_dataset_pb2.MCDataset()
            dataset.ParseFromString(f.read())
            self.embeddings = []
            self.raw = []
            self.measurements = []
            for d in dataset.data:
                nodes = []
                for n in d.graph.nodes:
                    one_hot = torch.zeros([d.graph.num_opcodes])
                    one_hot[n.onehot] = 1
                    nodes.append(one_hot)
                nodes = torch.stack(nodes)
                edges = []
                for e in d.graph.edges:
                    edges.append(torch.tensor([getattr(e, 'from', 0), e.to]))
                if len(edges) == 0:
                    continue
                edges = torch.stack(edges).t().contiguous()

                self.num_opcodes = d.graph.num_opcodes
                self.has_virtual_root = d.graph.has_virtual_root

                self.embeddings.append(Data(x=nodes, edge_index=edges))

                # TODO(Alex): use num_runs
                self.measurements.append(d.metrics.measured_cycles / 1000)

                raw = {"source": str(d.graph.source),
                       #"edges": d.graph.edges,
                       #"nodes": d.graph.nodes
                       }
                self.raw.append(raw)

    def len(self):
        return len(self.embeddings)

    def get(self, index):
        x = self.embeddings[index]
        y = self.measurements[index]
        z = self.raw[index]

        return x, y, z


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
