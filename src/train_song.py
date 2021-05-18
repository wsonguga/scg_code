import torch
import os, sys
from utils import get_data, Model, train_model, test_model
from tqdm import tqdm


NUM_LABELS = 4
HIDDEN_SIZE = 1024
NUM_LAYERS = 6
NUM_EPOCHS = 10
LR = 1e-3
WD = 1e-7
P = 0.1


if __name__ == "__main__":
    train_data_path = test_data_path = "../data/real_regression_data"
    out_path = "../outputs/real_regression_data"
    train_file_name = "real_train_timesorted"
    test_file_name = "real_test_timesorted"

    if(len(sys.argv) > 2):
        out_path = sys.argv[1]

        file_path = sys.argv[2]
        train_file_name = os.path.splitext(os.path.basename(file_path))[0]
        train_data_path = os.path.dirname(file_path)

        file_path = sys.argv[3]
        test_file_name = os.path.splitext(os.path.basename(file_path))[0]
        test_data_path = os.path.dirname(file_path)
    else:
        print(f"Usage: {sys.argv[0]} model_directory train_data_file test_data_file")
        print(f"Usage: {sys.argv[0]} ../outputs/song ../data/real_regression_data/real_train_truesorted.npy ../data/real_regression_data/real_test_truesorted.npy ")
        print(f"Usage: {sys.argv[0]} ../outputs/real_regression_data ../data/real_regression_data/real_train_timesorted.npy ../data/real_regression_data/real_test_timesorted.npy ")    
        exit()

    print(f"Input: {train_data_path+train_file_name} {test_data_path+test_file_name}")
    print(f"Output: {out_path}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_x, train_y = get_data(
        data_path=train_data_path, #data_path,
        out_path=out_path,
        name=train_file_name, #"real_train_timesorted",
        load_values=False,
        device=device,
        num_labels=NUM_LABELS,
        return_extra=False,
        drop_last = False,
        drop_extra=2)
    test_x, test_y, test_Y, min_v, max_v = get_data(
        data_path=test_data_path, #data_path,
        out_path=out_path,
        name=test_file_name, #"real_test_timesorted",
        load_values=True,
        device=device,
        num_labels=NUM_LABELS,
        return_extra=True,
        drop_last = False,
        drop_extra=2)

    model = Model(
        input_size=train_x.shape[1],
        num_labels=NUM_LABELS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        p=P).to(device)

    criterion = torch.nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 1000, 1e-4)

    for i in tqdm(range(NUM_EPOCHS), total=NUM_EPOCHS):
        train_loss = train_model(model, optim, criterion, train_x, train_y,
                                 scheduler=scheduler)
        if (i+1) % 100 == 0:
            test_losses = test_model(model, test_x, test_Y, min_v, max_v)
            tqdm.write(f"Epoch {i+1}/{NUM_EPOCHS}")
            tqdm.write(f"Train loss = {train_loss:.3f}")
            tqdm.write(f"MAE of H = {test_losses[0]:.3f}")
            tqdm.write(f"MAE of R = {test_losses[1]:.3f}")
            tqdm.write(f"MAE of S = {test_losses[2]:.3f}")
            tqdm.write(f"MAE of D = {test_losses[3]:.3f}")
            tqdm.write("")

            # model = model.to("cpu")
            # torch.save(model.state_dict(), f"{out_path}/model.pt")

    model = model.to("cpu")
    torch.save(model.state_dict(), f"{out_path}/model.pt")
