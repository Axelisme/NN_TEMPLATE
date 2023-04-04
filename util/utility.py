
"""some tools for the project"""

def set_seed(seed: int) -> None:
    """set seed for reproducibility"""
    import torch
    from torch.backends import cudnn
    import random
    import numpy as np

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

class Statistic():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        """reset the value"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update the value:
        input: the value to update, the number of the value"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Result:
    """handle the training result"""
    def __init__(self, results:dict = {}) -> None:
        """input: a dict of result, key is the name of the result, value is the result or a list of result"""
        self.data:dict = dict()
        self.log(results)

    def log(self,results:dict) -> None:
        """add a result:
        input: a dict of result, key is the name of the result, value is the result or a list of result"""
        for key,value in results.items():
            if key not in self.data.keys():
                self.data[key] = list()
            if isinstance(value, list):
                self.data[key].extend(value)
            else:
                self.data[key].append(value)

    def __getitem__(self, key) -> list:
        """get a result:
        input: the name of the result,
        output: a list of result"""
        return self.data[key]

    def save(self, path: str) -> None:
        """save the result to a csv file:
        input: the path to save the result"""
        import pandas as pd
        df = pd.DataFrame(self.data)
        df.to_csv(path,index=False)

    def load(self, path: str) -> None:
        """load the result from a csv file:
        input: the path to load the result"""
        import pandas as pd
        df = pd.read_csv(path)
        self.data = df.to_dict('list')

    def plot(self) -> None:
        """plot the result"""
        import matplotlib.pyplot as plt
        for key in self.data.keys():
            plt.plot(self.data[key], label=key)
        plt.legend()
        plt.show()