import torch


def TrainDataset(xy, track=50):
    #Choosing to use the GPU as the default processor for the ML model. Use the CPU if its not available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x0 = torch.from_numpy(xy[:,1:])
    n_samples = xy.shape[0]

def