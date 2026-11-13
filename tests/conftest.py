"""Small numerical tests avoid large host thread-pool launch overhead."""
import torch


def pytest_sessionstart(session):
    torch.set_num_threads(1)
