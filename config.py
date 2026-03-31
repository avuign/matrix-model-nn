import argparse

HIDDEN_DIM = 256
N_LAYERS = 6

GRID_PTS = 1000

FILLING_FRAC = 0.2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--g4", type=float, default=0.0)
    parser.add_argument("--n_epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--L", type=float, default=1)
    args = parser.parse_args()
    return args


WEIGHT_PATH = (
    f"saved_weights/m_" + str(get_args().m) + "_g4_" + str(get_args().g4) + ".pt"
)
FIG_PATH = f"images/m_" + str(get_args().m) + "_g4_" + str(get_args().g4) + ".png"
