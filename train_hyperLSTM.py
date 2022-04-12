from models.hyperLSTM import HyperLSTM


def run():
    # how many features are we using?
    INPUT_SIZE = 22

    # how many games do we want to store memories of?
    N_LAYERS = 10

    network = HyperLSTM(
        input_size=INPUT_SIZE,
        hidden_size=50,
        hyper_size=8,
        n_z=INPUT_SIZE,
        n_layers=N_LAYERS
    )

    


if __name__ == "__main__":
    run()
