import torch
from torch.nn import Parameter, ParameterList
import torch.nn as nn
import torch.nn.functional as F
import math


class myGRUCell(nn.Module):
    """
    wRank = rank of W matrix
    (creates 4 matrices if not None else creates 3 matrices)
    uRank = rank of U matrix
    (creates 4 matrices if not None else creates 3 matrices)

    GRU architecture and compression techniques are found in
    GRU(LINK) paper

    Basic architecture is like:

    r_t = gate_nl(W1x_t + U1h_{t-1} + B_r)
    z_t = gate_nl(W2x_t + U2h_{t-1} + B_g)
    h_t^ = update_nl(W3x_t + r_t*U3(h_{t-1}) + B_h)
    h_t = z_t*h_{t-1} + (1-z_t)*h_t^

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    """

    def __init__(self, input_size, hidden_size, wRank=None, uRank=None, recurrent_init=None,
                 hidden_init=None):
        super(myGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRank = uRank

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRank is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))

        self.bias_r = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, h):
        if self.wRank is None:
            wVal1 = torch.matmul(x, self.W1)
            wVal2 = torch.matmul(x, self.W2)
            wVal3 = torch.matmul(x, self.W3)
        else:
            wVal1 = torch.matmul(
                torch.matmul(x, self.W), self.W1)
            wVal2 = torch.matmul(
                torch.matmul(x, self.W), self.W2)
            wVal3 = torch.matmul(
                torch.matmul(x, self.W), self.W3)

        if self.uRank is None:
            uVal1 = torch.matmul(h, self.U1)
            uVal2 = torch.matmul(h, self.U2)
        else:
            uVal1 = torch.matmul(
                torch.matmul(h, self.U), self.U1)
            uVal2 = torch.matmul(
                torch.matmul(h, self.U), self.U2)
        matVal_r = wVal1 + uVal1
        matVal_z = wVal2 + uVal2

        r = F.sigmoid(matVal_r + self.bias_r)
        z = F.sigmoid(matVal_z + self.bias_gate)

        if self.uRank is None:
            matVal_c = wVal3 + torch.matmul(r * h, self.U3)
        else:
            matVal_c = wVal3 + \
                       torch.matmul(torch.matmul(r * h, self.U), self.U3)

        c_tilda = F.tanh(matVal_c + self.bias_update)

        h_next = z * h + (1.0 - z) * c_tilda

        return h_next


class myGRUCell_diag(nn.Module):
    """
    wRank = rank of W matrix
    (creates 4 matrices if not None else creates 3 matrices)
    uRank = rank of U matrix
    (creates 4 matrices if not None else creates 3 matrices)

    GRU architecture and compression techniques are found in
    GRU(LINK) paper

    Basic architecture is like:

    r_t = gate_nl(W1x_t + U1h_{t-1} + B_r)
    z_t = gate_nl(W2x_t + U2h_{t-1} + B_g)
    h_t^ = update_nl(W3x_t + r_t*U3(h_{t-1}) + B_h)
    h_t = z_t*h_{t-1} + (1-z_t)*h_t^

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    """

    def __init__(self, input_size, hidden_size, wRank=None, uRank=None, recurrent_init=None,
                 hidden_init=None):
        super(myGRUCell_diag, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRank = uRank

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRank is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U1_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            self.U2_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            self.U3_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))

        self.bias_r = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, h):
        if self.wRank is None:
            wVal1 = torch.matmul(x, self.W1)
            wVal2 = torch.matmul(x, self.W2)
            wVal3 = torch.matmul(x, self.W3)
        else:
            wVal1 = torch.matmul(
                torch.matmul(x, self.W), self.W1)
            wVal2 = torch.matmul(
                torch.matmul(x, self.W), self.W2)
            wVal3 = torch.matmul(
                torch.matmul(x, self.W), self.W3)

        if self.uRank is None:
            uVal1 = torch.matmul(h, self.U1)
            uVal2 = torch.matmul(h, self.U2)
        else:
            uVal1 = torch.matmul(
                torch.matmul(h, self.U), self.U1)
            uVal2 = torch.matmul(
                torch.matmul(h, self.U), self.U2)
            uVal1_diag = h * self.U1_diag
            uVal2_diag = h * self.U2_diag

        matVal_r = wVal1 + uVal1 + uVal1_diag
        matVal_z = wVal2 + uVal2 + uVal2_diag

        r = F.sigmoid(matVal_r + self.bias_r)
        z = F.sigmoid(matVal_z + self.bias_gate)

        if self.uRank is None:
            matVal_c = wVal3 + torch.matmul(r * h, self.U3)
        else:
            matVal_c = wVal3 + \
                       torch.matmul(torch.matmul(r * h, self.U), self.U3) + r * h * self.U3_diag
        c_tilda = F.tanh(matVal_c + self.bias_update)

        h_next = z * h + (1.0 - z) * c_tilda

        return h_next


class myLSTMCell(nn.Module):
    '''
    LR - Low Rank
    LSTM LR Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_nonlinearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_nonlinearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of all W matrices
    (creates 5 matrices if not None else creates 4 matrices)
    uRank = rank of all U matrices
    (creates 5 matrices if not None else creates 4 matrices)

    LSTM architecture and compression techniques are found in
    LSTM paper

    Basic architecture:

    f_t = gate_nl(W1x_t + U1h_{t-1} + B_f)
    i_t = gate_nl(W2x_t + U2h_{t-1} + B_i)
    C_t^ = update_nl(W3x_t + U3h_{t-1} + B_c)
    o_t = gate_nl(W4x_t + U4h_{t-1} + B_o)
    C_t = f_t*C_{t-1} + i_t*C_t^
    h_t = o_t*update_nl(C_t)

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    '''

    def __init__(self, input_size, hidden_size, wRank=None, uRank=None, recurrent_init=None,
                 hidden_init=None):
        super(myLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRank = uRank

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W4 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W4 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRank is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U4 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U4 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))

        self.bias_f = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_i = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_c = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_o = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, hiddenStates):
        (h, c) = hiddenStates

        if self.wRank is None:
            wVal1 = torch.matmul(x, self.W1)
            wVal2 = torch.matmul(x, self.W2)
            wVal3 = torch.matmul(x, self.W3)
            wVal4 = torch.matmul(x, self.W4)
        else:
            wVal1 = torch.matmul(
                torch.matmul(x, self.W), self.W1)
            wVal2 = torch.matmul(
                torch.matmul(x, self.W), self.W2)
            wVal3 = torch.matmul(
                torch.matmul(x, self.W), self.W3)
            wVal4 = torch.matmul(
                torch.matmul(x, self.W), self.W4)

        if self.uRank is None:
            uVal1 = torch.matmul(h, self.U1)
            uVal2 = torch.matmul(h, self.U2)
            uVal3 = torch.matmul(h, self.U3)
            uVal4 = torch.matmul(h, self.U4)
        else:
            uVal1 = torch.matmul(
                torch.matmul(h, self.U), self.U1)
            uVal2 = torch.matmul(
                torch.matmul(h, self.U), self.U2)
            uVal3 = torch.matmul(
                torch.matmul(h, self.U), self.U3)
            uVal4 = torch.matmul(
                torch.matmul(h, self.U), self.U4)
        matVal_i = wVal1 + uVal1
        matVal_f = wVal2 + uVal2
        matVal_o = wVal3 + uVal3
        matVal_c = wVal4 + uVal4

        i = F.sigmoid(matVal_i + self.bias_i)
        f = F.sigmoid(matVal_f + self.bias_f)
        o = F.sigmoid(matVal_o + self.bias_o)

        c_tilda = F.tanh(matVal_c + self.bias_c)

        c_next = f * c + i * c_tilda
        h_next = o * F.tanh(c_next)
        return c_next, h_next


class myLSTMCell_diag(nn.Module):
    '''
    LR - Low Rank
    LSTM LR Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_nonlinearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_nonlinearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of all W matrices
    (creates 5 matrices if not None else creates 4 matrices)
    uRank = rank of all U matrices
    (creates 5 matrices if not None else creates 4 matrices)

    LSTM architecture and compression techniques are found in
    LSTM paper

    Basic architecture:

    f_t = gate_nl(W1x_t + U1h_{t-1} + B_f)
    i_t = gate_nl(W2x_t + U2h_{t-1} + B_i)
    C_t^ = update_nl(W3x_t + U3h_{t-1} + B_c)
    o_t = gate_nl(W4x_t + U4h_{t-1} + B_o)
    C_t = f_t*C_{t-1} + i_t*C_t^
    h_t = o_t*update_nl(C_t)

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    '''

    def __init__(self, input_size, hidden_size, wRank=None, uRank=None, recurrent_init=None,
                 hidden_init=None):
        super(myLSTMCell_diag, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRank = uRank

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W4 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W4 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRank is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U4 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U4 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U1_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            self.U2_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            self.U3_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            self.U4_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))

        self.bias_f = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_i = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_c = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_o = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, hiddenStates):
        (h, c) = hiddenStates

        if self.wRank is None:
            wVal1 = torch.matmul(x, self.W1)
            wVal2 = torch.matmul(x, self.W2)
            wVal3 = torch.matmul(x, self.W3)
            wVal4 = torch.matmul(x, self.W4)
        else:
            wVal1 = torch.matmul(
                torch.matmul(x, self.W), self.W1)
            wVal2 = torch.matmul(
                torch.matmul(x, self.W), self.W2)
            wVal3 = torch.matmul(
                torch.matmul(x, self.W), self.W3)
            wVal4 = torch.matmul(
                torch.matmul(x, self.W), self.W4)

        if self.uRank is None:
            uVal1 = torch.matmul(h, self.U1)
            uVal2 = torch.matmul(h, self.U2)
            uVal3 = torch.matmul(h, self.U3)
            uVal4 = torch.matmul(h, self.U4)
        else:
            uVal1 = torch.matmul(
                torch.matmul(h, self.U), self.U1)
            uVal2 = torch.matmul(
                torch.matmul(h, self.U), self.U2)
            uVal3 = torch.matmul(
                torch.matmul(h, self.U), self.U3)
            uVal4 = torch.matmul(
                torch.matmul(h, self.U), self.U4)
            uVal1_diag = h * self.U1_diag
            uVal2_diag = h * self.U2_diag
            uVal3_diag = h * self.U3_diag
            uVal4_diag = h * self.U4_diag

        matVal_i = wVal1 + uVal1 + uVal1_diag
        matVal_f = wVal2 + uVal2 + uVal2_diag
        matVal_o = wVal3 + uVal3 + uVal3_diag
        matVal_c = wVal4 + uVal4 + uVal4_diag

        i = F.sigmoid(matVal_i + self.bias_i)
        f = F.sigmoid(matVal_f + self.bias_f)
        o = F.sigmoid(matVal_o + self.bias_o)

        c_tilda = F.tanh(matVal_c + self.bias_c)

        c_next = f * c + i * c_tilda
        h_next = o * F.tanh(c_next)
        return c_next, h_next


class myGRU(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRank=8, **kwargs):
        super(myGRU, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank = wRank
        self.uRank = uRank

        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1

        rnn_cells = []
        in_size = input_size
        i = 0
        for i, hidden_size in enumerate(hidden_layer_sizes):
            if recurrent_inits is not None:
                kwargs["recurrent_init"] = recurrent_inits[i]
            if hidden_inits is not None:
                kwargs["hidden_init"] = hidden_inits[i]
            rnn_cells.append(myGRUCell(in_size, hidden_size, wRank=self.wRank, uRank=self.uRank, **kwargs))
            in_size = hidden_size

        self.rnncells = nn.ModuleList(rnn_cells)

        # h0 = torch.zeros(hidden_size * num_directions, requires_grad=False)
        # self.register_buffer('h0', h0)

    def forward(self, x, hidden=None):
        time_index = self.time_index
        batch_index = self.batch_index
        hiddens = []

        i = 0
        for cell in self.rnncells:
            # hx = self.h0.unsqueeze(0).expand(
            #    x.size(batch_index),
            #    self.hidden_size * num_directions).contiguous()
            self.device = x.device
            hx = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            x_n = []
            outputs = []
            x_time = torch.unbind(x, time_index)
            seqlen = len(x_time)
            for t in range(seqlen):
                hx = cell(x_time[t], hx)
                outputs.append(hx)
            x = torch.stack(outputs, time_index)
            # x=torch.cat(outputs, -1)
            hiddens.append(hx)
            i = i + 1

        return x, torch.cat(hiddens, -1)


class myGRU2(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRank=8, **kwargs):
        super(myGRU2, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank = wRank
        self.uRank = uRank

        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1

        rnn_cells = []
        in_size = input_size
        i = 0
        for i, hidden_size in enumerate(hidden_layer_sizes):
            if recurrent_inits is not None:
                kwargs["recurrent_init"] = recurrent_inits[i]
            if hidden_inits is not None:
                kwargs["hidden_init"] = hidden_inits[i]
            rnn_cells.append(myGRUCell_diag(in_size, hidden_size, wRank=self.wRank, uRank=self.uRank, **kwargs))
            in_size = hidden_size

        self.rnncells = nn.ModuleList(rnn_cells)

        # h0 = torch.zeros(hidden_size * num_directions, requires_grad=False)
        # self.register_buffer('h0', h0)

    def forward(self, x, hidden=None):
        time_index = self.time_index
        batch_index = self.batch_index
        hiddens = []

        i = 0
        for cell in self.rnncells:
            # hx = self.h0.unsqueeze(0).expand(
            #    x.size(batch_index),
            #    self.hidden_size * num_directions).contiguous()
            self.device = x.device
            hx = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            x_n = []
            outputs = []
            x_time = torch.unbind(x, time_index)
            seqlen = len(x_time)
            for t in range(seqlen):
                hx = cell(x_time[t], hx)
                outputs.append(hx)
            x = torch.stack(outputs, time_index)
            # x=torch.cat(outputs, -1)
            hiddens.append(hx)
            i = i + 1

        return x, torch.cat(hiddens, -1)


class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRank=8, **kwargs):
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank = wRank
        self.uRank = uRank

        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1

        rnn_cells = []
        in_size = input_size
        i = 0
        for i, hidden_size in enumerate(hidden_layer_sizes):
            if recurrent_inits is not None:
                kwargs["recurrent_init"] = recurrent_inits[i]
            if hidden_inits is not None:
                kwargs["hidden_init"] = hidden_inits[i]
            rnn_cells.append(myLSTMCell(in_size, hidden_size, wRank=self.wRank, uRank=self.uRank, **kwargs))
            in_size = hidden_size

        self.rnncells = nn.ModuleList(rnn_cells)

        # h0 = torch.zeros(hidden_size * num_directions, requires_grad=False)
        # self.register_buffer('h0', h0)

    def forward(self, x, hidden=None):
        time_index = self.time_index
        batch_index = self.batch_index
        hiddens = []

        i = 0
        for cell in self.rnncells:
            # hx = self.h0.unsqueeze(0).expand(
            #    x.size(batch_index),
            #    self.hidden_size * num_directions).contiguous()
            self.device = x.device
            h = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            c = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            x_n = []
            outputs = []
            x_time = torch.unbind(x, time_index)
            seqlen = len(x_time)
            for t in range(seqlen):
                h, c = cell(x_time[t], (h, c))
                outputs.append(h)
            x = torch.stack(outputs, time_index)
            # x=torch.cat(outputs, -1)
            hiddens.append(h)
            i = i + 1

        return x, torch.cat(hiddens, -1)


class myLSTM2(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRank=8, **kwargs):
        super(myLSTM2, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank = wRank
        self.uRank = uRank

        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1

        rnn_cells = []
        in_size = input_size
        i = 0
        for i, hidden_size in enumerate(hidden_layer_sizes):
            if recurrent_inits is not None:
                kwargs["recurrent_init"] = recurrent_inits[i]
            if hidden_inits is not None:
                kwargs["hidden_init"] = hidden_inits[i]
            rnn_cells.append(myLSTMCell_diag(in_size, hidden_size, wRank=self.wRank, uRank=self.uRank, **kwargs))
            in_size = hidden_size

        self.rnncells = nn.ModuleList(rnn_cells)

        # h0 = torch.zeros(hidden_size * num_directions, requires_grad=False)
        # self.register_buffer('h0', h0)

    def forward(self, x, hidden=None):
        time_index = self.time_index
        batch_index = self.batch_index
        hiddens = []

        i = 0
        for cell in self.rnncells:
            # hx = self.h0.unsqueeze(0).expand(
            #    x.size(batch_index),
            #    self.hidden_size * num_directions).contiguous()
            self.device = x.device
            h = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            c = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            x_n = []
            outputs = []
            x_time = torch.unbind(x, time_index)
            seqlen = len(x_time)
            for t in range(seqlen):
                h, c = cell(x_time[t], (h, c))
                outputs.append(h)
            x = torch.stack(outputs, time_index)
            # x=torch.cat(outputs, -1)
            hiddens.append(h)
            i = i + 1

        return x, torch.cat(hiddens, -1)

