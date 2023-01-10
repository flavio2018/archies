import torch
from .LSTM import DeepLSTM2


class GatingNetwork(torch.nn.Module):
    
    def __init__(self, hidden_size, num_experts, topk):
        super(GatingNetwork, self).__init__()
        self.linear = torch.nn.Linear(in_features=hidden_size,
                                      out_features=num_experts)
        self.softmax = torch.nn.Softmax(dim=1)
        self.topk = topk
    
    def forward(self, x):
        x = self.softmax(self.linear(x))
        return torch.topk(x, dim=1, k=self.topk)


class RecurrentDispatcher(torch.nn.Module):
    
    def __init__(self, lstm, gating_net):
        super(RecurrentDispatcher, self).__init__()
        self.lstm = lstm
        self.gating_net = gating_net
        
    def forward(self, x, valid_pos):
        _ = self.lstm.step(x, valid_pos)
        return self.gating_net(self.lstm.c_t_2)
    
    def get_states(self):
        return self.lstm.get_states()
    
    def set_states(self, states):
        self.lstm.set_states(states)


class RecurrentMoE(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_experts, topk):
        super(RecurrentMoE, self).__init__()
        self.recurrent_dispatcher = RecurrentDispatcher(DeepLSTM2(input_size,
                                                                 2*[hidden_size],
                                                                 output_size,
                                                                 batch_size),
                                                        GatingNetwork(hidden_size,
                                                                      num_experts,
                                                                      topk))
        self.experts = torch.nn.ModuleList(num_experts*[DeepLSTM2(input_size,
                                                                   2*[hidden_size],
                                                                   output_size,
                                                                   batch_size)])
    
    def step(self, x, valid_pos=None):
        if valid_pos is None:  # all positions are valid
            valid_pos = torch.tensor(range(x.size(0)), device=x.device)
        output_batch = torch.zeros_like(x)
        experts_weights, selected_experts = self.recurrent_dispatcher(x, valid_pos)
        experts_weights, selected_experts = experts_weights[valid_pos], selected_experts[valid_pos]
        
        for i_e, expert in enumerate(self.experts):
            tokens_for_expert = torch.where(selected_experts==i_e)[0]
            weights_tokens_for_expert = experts_weights[torch.where(selected_experts==i_e)]
            output = expert.step(x, tokens_for_expert)[tokens_for_expert]
            weighted_output = weights_tokens_for_expert.unsqueeze(1) * output
            output_batch[tokens_for_expert] = output_batch[tokens_for_expert] + weighted_output
        return output_batch
    
    def forward(self, x, lengths=None):
        outputs = []
        
        if lengths is not None:
            non_padded_positions = torch.argwhere(lengths > 0).squeeze()

            for t in range(x.size(1)):
                o = self.step(x[:, t, :], non_padded_positions)
                lengths = self._reduce_lengths(lengths)
                non_padded_positions = torch.argwhere(lengths > 0).squeeze()
                outputs.append(o.unsqueeze(1))
        else:
            for t in range(x.size(1)):
                if outputs:
                    o = self.step(o)
                else:
                    o = self.step(x[:, t, :])
                outputs.append(o.unsqueeze(1))
        return torch.concat(outputs, dim=1)

    def _reduce_lengths(self, lengths):
        return torch.where(lengths > 0, lengths - 1, 0)


class EncoderDecoderRecurrentMoE(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_experts, topk):
        super(EncoderDecoderRecurrentMoE, self).__init__()
        self.encoder = RecurrentMoE(input_size, hidden_size, output_size, num_experts, topk)
        self.decoder = RecurrentMoE(input_size, hidden_size, output_size, num_experts, topk)
    
    def forward(self, x, y, lengths):
        enc_out = self.encoder(x, lengths)
        
        enc_disp_states = self.encoder.recurrent_dispatcher.get_states()
        self.decoder.recurrent_dispatcher.set_states(enc_disp_states)
        for enc_expert, dec_expert in zip(self.encoder.experts, self.decoder.experts):
            dec_expert.set_states(enc_expert.get_states())
        
        dec_out = self.decoder(y)
        return dec_out
