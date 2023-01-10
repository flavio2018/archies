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
