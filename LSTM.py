class DeepLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=16):
        super(DeepLSTM, self).__init__()
        self.lstm_cell_1 = torch.nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size[0])
        self.lstm_cell_2 = torch.nn.LSTMCell(
            input_size=hidden_size[0],
            hidden_size=hidden_size[1])
        self.W_o = torch.nn.Parameter(torch.rand(output_size, hidden_size[0]+hidden_size[1])*0.01)
        self.b_o = torch.nn.Parameter(torch.rand(1)*0.01)
        self.register_buffer("h_t_1", torch.zeros(batch_size, hidden_size[0]))
        self.register_buffer("c_t_1", torch.zeros(batch_size, hidden_size[0]))
        self.register_buffer("h_t_2", torch.zeros(batch_size, hidden_size[1]))
        self.register_buffer("c_t_2", torch.zeros(batch_size, hidden_size[1]))
        self._init_parameters()
    
    def _init_parameters(self):
        for name, param in self.lstm_cell_1.named_parameters():
            if 'bias' in name:
                torch.nn.init.normal_(param, std=0.1)
            else:
                torch.nn.init.xavier_normal_(param, gain=torch.nn.init.calculate_gain('tanh', param))

        for name, param in self.lstm_cell_2.named_parameters():
            if 'bias' in name:
                torch.nn.init.normal_(param, std=0.1)
            else:
                torch.nn.init.xavier_normal_(param, gain=torch.nn.init.calculate_gain('tanh', param))
    
    def forward(self, x):
        self.h_t_1, self.c_t_1 = self.lstm_cell_1(x, (self.h_t_1, self.c_t_1))
        self.h_t_2, self.c_t_2 = self.lstm_cell_2(self.h_t_1, (self.h_t_2, self.c_t_2))
        # self.h_t_2 + self.h_t_1  # skip connection
        self.h_t_12 = torch.concat([self.h_t_1, self.h_t_2], dim=1)  # layer-wise outputs concat
        return self.h_t_12 @ self.W_o.T + self.b_o

    def detach_states(self):
        self.h_t_1 = self.h_t_1.detach()
        self.c_t_1 = self.c_t_1.detach()
        self.h_t_2 = self.h_t_2.detach()
        self.c_t_2 = self.c_t_2.detach()
        self.h_t_1.fill_(0)
        self.c_t_1.fill_(0)
        self.h_t_2.fill_(0)
        self.c_t_2.fill_(0)
    
    def set_states(self, h_dict, c_dict):
        # make lists to sort
        h_1 = [(i, h) for i, h in h_dict[1].items()]
        h_2 = [(i, h) for i, h in h_dict[2].items()]
        c_1 = [(i, c) for i, c in c_dict[1].items()]
        c_2 = [(i, c) for i, c in c_dict[2].items()]
        
        # sort
        h_1 = sorted(h_1, key=lambda x: x[0])
        h_2 = sorted(h_2, key=lambda x: x[0])
        c_1 = sorted(c_1, key=lambda x: x[0])
        c_2 = sorted(c_2, key=lambda x: x[0])
        
        self.h_t_1 = torch.concat([h for _, h in h_1])
        self.h_t_2 = torch.concat([h for _, h in h_2])
        self.c_t_1 = torch.concat([c for _, c in c_1])
        self.c_t_2 = torch.concat([c for _, c in c_2])


class DeepLSTM2(DeepLSTM):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(DeepLSTM2, self).__init__(input_size, hidden_size, output_size, batch_size)

    def step(self, x, valid_pos=None):
        """Process a slice corresponding to a single timestep,
        updating only non-padded positions of the cells."""
        if valid_pos is None:  # all positions are valid
            valid_pos = torch.tensor(range(x.size(0)), device=x.device)
        self.h_t_1[valid_pos], self.c_t_1[valid_pos] = self.lstm_cell_1(x[valid_pos], (self.h_t_1[valid_pos], self.c_t_1[valid_pos]))
        self.h_t_2[valid_pos], self.c_t_2[valid_pos] = self.lstm_cell_2(self.h_t_1[valid_pos], (self.h_t_2[valid_pos], self.c_t_2[valid_pos]))
        # self.h_t_2 + self.h_t_1  # skip connection
        self.h_t_12 = torch.concat([self.h_t_1, self.h_t_2], dim=1)  # layer-wise outputs concat
        return self.h_t_12 @ self.W_o.T + self.b_o

    def forward(self, x, lengths=None):
        """Process a batch of sequences of vectors (BxLxD)."""
        outputs = []
        
        if lengths is not None:  # used as encoder on inputs
            non_padded_positions = torch.argwhere(lengths > 0).squeeze()

            for t in range(x.size(1)):
                o = self.step(x[:, t, :].squeeze(), non_padded_positions)
                lengths = self._reduce_lengths(lengths)
                non_padded_positions = torch.argwhere(lengths > 0).squeeze()
                outputs.append(o.unsqueeze(1))  # add time dim
        else:  # used as decoder to generate outputs
            for t in range(x.size(1)):
                if outputs:
                    o = self.step(o)  # no TF, step on previous output
                else:
                    o = self.step(x[:, t, :])
                outputs.append(o.unsqueeze(1))
        return torch.concat(outputs, dim=1)

    def set_states(self, states):
        h1, h2, c1, c2 = states
        self.h_t_1, self.h_t_2 = h1, h2
        self.c_t_1, self.c_t_2 = c1, c2

    def get_states(self):
        return self.h_t_1, self.h_t_2, self.c_t_1, self.c_t_2

    def _reduce_lengths(self, lengths):
        return torch.where(lengths > 0, lengths - 1, 0)
