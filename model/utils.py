import os
import json
import torch
import yaml
from torch_geometric.data import Data, Dataset


def load_basic_block_data(json_file_path):
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    # Convert node features into a tensor
    node_features = []
    for node in data["nodes"]:
        node_feature = [
            node["is_barrier"],
            node["is_compute"],
            node["is_load"],
            node["is_store"],
            node["is_vector"],
            node["opcode"],
        ]
        node_features.append(node_feature)
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Convert edges into a tensor
    edge_index = torch.tensor(data["edges"], dtype=torch.long).t().contiguous()

    # Create a PyTorch Geometric Data object
    graph_data = Data(x=node_features, edge_index=edge_index)

    return graph_data, data


def load_all_basic_blocks_data(directory):
    basic_blocks_data = []
    raw = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".json"):
            json_file_path = os.path.join(directory, file_name)
            basic_block_data, source = load_basic_block_data(json_file_path)
            basic_blocks_data.append(basic_block_data)
            raw.append(source)

    return basic_blocks_data, raw


def load_measured_data(directory):
    measured_cycles = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".yaml"):
            file_path = os.path.join(directory, file_name)
            file = open(file_path, "r")
            data = yaml.safe_load(file)
            file.close()
            measured_cycles.append(float(data["results"]["cycles"]) / data["results"]["num_runs"])
    return measured_cycles


class BasicBlockDataset(Dataset):
    def __init__(self, embeddings_path, measurements_path):
        super().__init__(None, None, None)
        embeddings, raw = load_all_basic_blocks_data(embeddings_path)
        self.embeddings = embeddings
        self.raw = raw
        self.measurements = load_measured_data(measurements_path)

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

    cpu_port_pressure = port_pressure_sequence.to("cpu").detach().numpy()

    for instruction in cpu_port_pressure:
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
