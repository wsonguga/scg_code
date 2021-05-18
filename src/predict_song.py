import torch
import sys, os
from utils import get_data, Model, test_model, sort_dataset

NUM_LABELS = 4
HIDDEN_SIZE = 1024
NUM_LAYERS = 3


if __name__ == "__main__":
    file_name = "real_test_timesorted"
    data_path = "../data/real_regression_data"
    out_path = "../outputs/real_regression_data"
    if(len(sys.argv) > 2):
        out_path = sys.argv[1]
        file_path = sys.argv[2]
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        data_path = os.path.dirname(file_path)

    else:
        print(f"Usage: {sys.argv[0]} model_directory test_data_file")
        print(f"Example: {sys.argv[0]} ../outputs/song ../data/real_regression_data/real_test_truesorted.npy")
        print(f"Example: {sys.argv[0]} ../outputs/real_regression_data ../data/real_regression_data/real_test_timesorted.npy")
        exit()

    print(f"Input: {data_path+file_name}")
    print(f"Output: {out_path}")

    device = torch.device('cpu')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # test_x, test_y, test_Y, min_v, max_v = get_data(
    #     data_path=data_path,
    #     out_path=out_path,
    #     name="real_test_timesorted",
    #     load_values=True,
    #     device=device,
    #     num_labels=NUM_LABELS,
    #     return_extra=True,
    #     drop_extra=2)
    
    test_x, test_y, test_Y, min_v, max_v = get_data(
        data_path=data_path,
        out_path=out_path,
        name=file_name,
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


# %%
    # sort_dataset()

# %%
