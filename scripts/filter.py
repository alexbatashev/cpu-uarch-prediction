import sys
import os
import json
import yaml

# USAGE: python filter.py /path/to/data /path/to/measured
# Filter out failed measures (e.g., ones with calls)

data_path = sys.argv[1]
measure_path = sys.argv[2]

measure_files = [file_name for file_name in os.listdir(measure_path)]
data_files = [file_name for file_name in os.listdir(data_path)]

# Remove invalid measurements
for m in measure_files:
    data = m.replace(".yaml", ".json")
    if not os.path.exists(os.path.join(data_path, data)):
        os.unlink(os.path.join(measure_path, m))
        continue
    with open(os.path.join(measure_path, m), "r") as f:
        res = yaml.safe_load(f)
        if res["results"]["total_cycles"] <= res["results"]["noise"]:
            os.unlink(os.path.join(measure_path, m))
        elif res["results"]["cycles"] / res["results"]["num_runs"] > 1000:
            # This is a suspicious run
            os.unlink(os.path.join(measure_path, m))

measure_files = [file_name for file_name in os.listdir(measure_path)]

for d in data_files:
    measure = d.replace(".json", ".yaml")
    if not os.path.exists(os.path.join(measure_path, measure)):
        os.unlink(os.path.join(data_path, d))
        continue

    #with open(os.path.join(data_path, d), "r") as json_file:
    #    data = json.load(json_file)
    #    if len(data["nodes"]) == 0:
    #        os.unlink(os.path.join(data_path, d))
    #        os.unlink(os.path.join(measure_path, measure))
