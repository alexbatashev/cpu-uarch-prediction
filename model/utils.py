import io
import numpy as np
import matplotlib.pyplot as plt


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


def plot_lift_chart(data):
    values = np.asarray(data)
    values = np.transpose(values)
    values = sorted(values, key=lambda d: d[0])
    values = np.transpose(values)

    num_samples = len(values[0, :])

    plt.figure()
    items = np.arange(0, num_samples)
    plt.plot(items, values[1, :], 'b', items, values[0, :], 'r')
    plt.xlim(0, num_samples)
    plt.xlabel('Item')
    plt.ylabel('measured vs predicted')
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf
