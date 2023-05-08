import model.utils

dataset = model.utils.BasicBlockDataset("./data/i5_1135g7.pb")
el = dataset[151]
print(el)
print(dataset.has_virtual_root)