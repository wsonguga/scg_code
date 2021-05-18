import torch
from utils import get_data, Model, test_model

data_path = "../data/real_regression_data"
out_path = "../outputs/real_regression_data"

NUM_LABELS = 4
HIDDEN_SIZE = 1024
NUM_LAYERS = 3


if __name__ == "__main__":
    device = torch.device('cpu')

    test_x, test_y, test_Y, min_v, max_v = get_data(
        data_path=data_path,
        out_path=out_path,
        name="real_test_timesorted",
        load_values=True,
        device=device,
        num_labels=NUM_LABELS,
        return_extra=True,
        drop_extra=2)

    model = Model(
        input_size=test_x.shape[1],
        num_labels=NUM_LABELS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load(f"{out_path}/model.pt"))
    model = model.to(device)
    model.eval()

    losses = test_model(model, test_x, test_Y, min_v, max_v)

    print(f"MAE of H = {losses[0]:.3f}")
    print(f"MAE of R = {losses[1]:.3f}")
    print(f"MAE of S = {losses[2]:.3f}")
    print(f"MAE of D = {losses[3]:.3f}")
