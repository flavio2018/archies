import torch


class CustomMLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: str, output_size: int):
        """hidden_sizes should be a list of comma separated integers."""
        hidden_sizes = [int(size) for size in hidden_sizes.split(",")]
        super().__init__()
        hidden_sizes = [input_size] + hidden_sizes
        self.linear_layers = []
        for i in range(len(hidden_sizes) - 1):
            self.linear_layers += [torch.nn.Linear(in_features=hidden_sizes[i],
                                                   out_features=hidden_sizes[i+1])]
        self.output_layer = torch.nn.Linear(in_features=hidden_sizes[-1], out_features=output_size)

    def forward(self, x):
        x = x.view(-1, 784)
        print(get_digit_string_repr(x[0]))
        relu = torch.nn.ReLU()

        for layer in self.linear_layers:
            x = layer(x)
            x = relu(x)
        x = self.output_layer(x).T  # output shape is (-1, output_size)
        return None, torch.nn.functional.log_softmax(x, dim=0)

    def prepare_for_batch(self, batch, device):
        return


def build_mlp(model_conf, device):
    return CustomMLP(input_size=model_conf.input_size,
                     hidden_sizes=model_conf.hidden_sizes,
                     output_size=model_conf.output_size).to(device)
