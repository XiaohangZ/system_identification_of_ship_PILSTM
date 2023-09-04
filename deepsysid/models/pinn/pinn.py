import torch.nn as nn
import torch
from torch.utils import data
from typing import Dict, List, Literal, Optional, Tuple
from numpy.typing import NDArray
import numpy as np


class PINNNet(nn.Module):
    def __init__(self, inputNode=2, hiddenNode=256, outputNode=1):
        super(PINNNet, self).__init__()
        # Define Hyperparameters
        self.inputLayerSize = inputNode
        self.outputLayerSize = outputNode
        self.hiddenLayerSize = hiddenNode
        # weights
        self.Linear1 = nn.Linear(self.inputLayerSize, self.hiddenLayerSize)
        self.Linear2 = nn.Linear(self.hiddenLayerSize, self.outputLayerSize)
        self.activation = torch.nn.Sigmoid()

    def forward(self, X):
        out1 = self.Linear1(X)
        out2 = self.activation(out1)
        out3 = self.Linear2(out2)
        return out3

class PILSTMNet(nn.Module):
    def __init__(self, inputNode=2, hiddenNode=256, outputNode=1):
        super(PILSTMNet, self).__init__()

        self.inp = nn.Linear(inputNode, hiddenNode)
        self.rnn = nn.LSTM(hiddenNode, hiddenNode, 2)  # input,hidden,layers
        # self.rnn = nn.RNN(hiddenNode , hiddenNode , 3)
        self.out = nn.Linear(hiddenNode, 1)

        self.outputNode = outputNode

    def step(self, input, hidden=None):
        input = self.inp(input).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, steps=40, hidden=None):
        # format: batch, seq, features
        outputs = torch.zeros(inputs.shape[0], inputs.shape[1], self.outputNode)

        for i in range(steps):
            input = inputs[:, i]

            # print(input.shape)
            output, hidden = self.step(input, hidden)

            # print(output.shape)
            outputs[:, i] = output

        return outputs

def pinn_loss(r_true, r_pred, delta, U, r_prv):
    T_dotr = 221/5
    T_U_dotr = -62/45
    T_U2_dotr = 449/180
    T_U3_dotr = -193/1620
    K_delta = -7/100
    K_U_delta = 1/360
    K_U2_delta = 1/180
    K_U3_delta = -1/3024
    N_r = 1
    N_r3 = 1/2
    N_U_r3 = -43/180
    N_U2_r3 = 1/18
    N_U3_r3 = -1/324
    sampling_time = 1
    assert r_true.shape == r_pred.shape == delta.shape == U.shape
    F_rudder = K_delta * delta + K_U_delta * U * delta + K_U2_delta * U*U * delta + K_U3_delta * U*U*U * delta
    F_hydro = N_r * r_pred + N_r3 * r_pred*r_pred*r_pred + N_U_r3 * U * r_pred*r_pred*r_pred + N_U2_r3 * U*U * r_pred*r_pred*r_pred + N_U3_r3 * U*U*U * r_pred*r_pred*r_pred
    r_dot = (r_pred - r_prv)/sampling_time
    R = F_rudder - F_hydro - (T_dotr + T_U_dotr * U + T_U2_dotr * U*U + T_U3_dotr * U*U*U) * r_dot
    return torch.mean(R*R)

class RecurrentPINNDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
            self,
            control_seqs: List[NDArray[np.float64]],
            state_seqs: List[NDArray[np.float64]],
            sequence_length: int,
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.x, self.y = self.__load_data(control_seqs, state_seqs)

    def __load_data(
            self,
            control_seqs: List[NDArray[np.float64]],
            state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        x_seq = list()
        y_seq = list()
        for control, state in zip(control_seqs, state_seqs):
            n_samples = int(
                (control.shape[0] - self.sequence_length - 1) / self.sequence_length
            )

            x = np.zeros(
                (n_samples, self.sequence_length, self.control_dim),
                dtype=np.float64,
            )
            y = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )

            for idx in range(n_samples):
                time = idx * self.sequence_length

                x[idx, :, :] = control[time: time + self.sequence_length, :]
                y[idx, :, :] = state[time: time + self.sequence_length, :]

            x_seq.append(x)
            y_seq.append(y)

        return np.vstack(x_seq), np.vstack(y_seq)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {'x': self.x[idx], 'y': self.y[idx]}

