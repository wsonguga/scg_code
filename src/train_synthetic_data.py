import torch
from utils import get_data, Model, train_model, test_model

data_path = "../data/synthetic_data"
out_path = "../outputs/synthetic_data"

NUM_LABELS = 3
HIDDEN_SIZE = 1024
NUM_LAYERS = 4
NUM_EPOCHS = 1000
LR = 0.0005
WD = 1e-5

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_x, train_y = get_data(
        data_path=data_path,
        out_path=out_path,
        name="synthetic_train_set",
        load_values=False,
        device=device,
        num_labels=NUM_LABELS,
        return_extra=False)
    test_x, test_y, test_Y, min_v, max_v = get_data(
        data_path=data_path,
        out_path=out_path,
        name="synthetic_test_set",
        load_values=True,
        device=device,
        num_labels=NUM_LABELS,
        return_extra=True)

    model = Model(
        input_size=train_x.shape[1],
        num_labels=NUM_LABELS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS).to(device)

    criterion = torch.nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    for i in range(NUM_EPOCHS):
        train_loss = train_model(model, optim, criterion, train_x, train_y)
        test_losses = test_model(model, test_x, test_Y, min_v,
                                 max_v)

        if (i+1) % 25 == 0:
            print(f"Epoch {i+1}/{NUM_EPOCHS}")
            print(f"Train loss = {train_loss:.3f}")
            print(f"MAE of S = {test_losses[0]:.3f}")
            print(f"MAE of D = {test_losses[1]:.3f}")
            print(f"MAE of H = {test_losses[2]:.3f}")
            print()

    model = model.to("cpu")
    torch.save(model.state_dict(), f"{out_path}/model.pt")
