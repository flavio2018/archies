import torch


class CustomLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, proj_size):
        super().__init__()
        self.lstm = torch.nn.LSTMCell(input_size=input_size,
                                      hidden_size=hidden_size)
        self.linear = torch.nn.Linear(in_features=hidden_size,
                                      out_features=proj_size)
        self.register_buffer("hidden_states", torch.zeros(1, hidden_size))
        self.register_buffer("cell_states", torch.zeros(1, hidden_size))
        self.hidden_size = hidden_size
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, batch):
        batch_size, sequence_len, feature_size = batch.shape
        all_outputs = []
        all_hidden_states = []
        all_cell_states = []

        for seq_pos in range(sequence_len):
            self.hidden_states, self.cell_states = self.lstm(batch[:, seq_pos, :].reshape(batch_size, feature_size),
                                                             (self.hidden_states, self.cell_states))
            o_n = self.log_softmax(self.linear(self.hidden_states))
            all_hidden_states.append(self.hidden_states)
            all_cell_states.append(self.cell_states)
            all_outputs.append(o_n.T)
        return (torch.stack(all_hidden_states), torch.stack(all_cell_states)), torch.stack(all_outputs)

    def prepare_for_batch(self, batch, device):
        batch_size, sequence_len, feature_size = batch.shape
        self.register_buffer("hidden_states", torch.zeros(batch_size, self.hidden_size, device=device))
        self.register_buffer("cell_states", torch.zeros(batch_size, self.hidden_size, device=device))

    def set_hidden_state(self, hn_cn, input_sequences_lengths, batch_size):
        hidden_states, cell_states = hn_cn
        self.hidden_states = torch.stack([hidden_states[l-1,b,:] for l, b in zip(input_sequences_lengths, range(batch_size))])
        self.cell_states = torch.stack([cell_states[l-1,b,:] for l, b in zip(input_sequences_lengths, range(batch_size))])
        return self.hidden_states


def build_lstm(model_conf, device):
    return CustomLSTM(input_size=model_conf.input_size,
                      hidden_size=model_conf.hidden_size,
                      proj_size=model_conf.output_size).to(device)
