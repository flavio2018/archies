import torch


class CustomRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, proj_size):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size=input_size,
                                hidden_size=hidden_size)
        self.output_mlp = torch.nn.Linear(in_features=hidden_size, out_features=proj_size)

    def forward(self, batch):
        full_output, h_n = self.rnn(batch)
        last_output = full_output[-1, :, :]  # select only the output for the last timestep and reshape to 2D
        projected_output = self.output_mlp(last_output).T
        log_soft_output = torch.nn.functional.log_softmax(projected_output, dim=0)

        # print(get_digit_string_repr(batch[:, 0, :]))
        # print(f"{full_output.shape=}")
        # print(f"{last_output.shape=}")
        # print(f"{projected_output.shape=}")
        # print(f"{log_soft_output.shape=}")

        return h_n, log_soft_output

    def prepare_for_batch(self, batch, device):
        return


def build_rnn(model_conf, device):
    return CustomRNN(input_size=model_conf.input_size,
                     hidden_size=model_conf.hidden_size,
                     proj_size=model_conf.output_size).to(device)
