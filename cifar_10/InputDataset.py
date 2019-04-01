from torch.utils.data import Dataset


class InputDataset(Dataset):
    """Helper class for input datasets"""

    def __init__(self, inputs, results, label, input_dim):

        self.inputs = inputs
        self.results = results
        self.input_dim = input_dim
        self.label = label

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        return {'InputVector': self.inputs[idx], 'Result': self.results[idx], 'Label': self.label[idx]}
