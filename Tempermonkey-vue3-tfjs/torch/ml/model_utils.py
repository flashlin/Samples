def reduce_dim(torch_value):
    return torch_value.squeeze()


def detach_lstm_hidden_state(states):
    return [state.detach() for state in states]
