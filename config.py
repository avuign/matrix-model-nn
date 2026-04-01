import argparse

HIDDEN_DIM = 512
N_LAYERS = 8

GRID_PTS = 2000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--g4", type=float, default=0.0)
    parser.add_argument("--g6", type=float, default=0.0)
    parser.add_argument("--g8", type=float, default=0.0)
    parser.add_argument("--n_epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--L", type=float, default=1)
    parser.add_argument("--nu", type=float, default=0.5)
    parser.add_argument("--plot_an", type=bool, default=False)

    args = parser.parse_args()
    return args


WEIGHT_PATH = (
    f"saved_weights/m_" + str(get_args().m) + "_g4_" + str(get_args().g4) + ".pt"
)
FIG_PATH = f"images/m_" + str(get_args().m) + "_g4_" + str(get_args().g4) + ".png"

FILLING_FRAC = get_args().nu
