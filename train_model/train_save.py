import torch
from pathlib import Path
from tqdm.auto import tqdm
from timeit import default_timer

def train_save_model(model: torch.nn.Module, model_save_name: str, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer:torch.optim.Optimizer, epochs: int, device, writer=None):
    start_training_time = default_timer()
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        test_loss, test_acc = 0, 0
        train_loss, train_acc = 0, 0

        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_class = torch.max(torch.softmax(y_pred, dim=1), dim=1).indices
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                test_pred = model(X)

                test_loss += loss_fn(test_pred, y).item()

                test_pred_labels = torch.max(test_pred, dim=1).indices
                test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)

        train_loss /= len(train_dataloader)
        test_loss /= len(test_dataloader)
        train_acc = train_acc * 100 / len(train_dataloader)
        test_acc = test_acc * 100 / len(test_dataloader)

        print(f"Epoch: {epoch}\n--------------------------------")
        print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.2f}%\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f}%\n--------------------------------")

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["train_acc"].append(train_acc)
        results["test_acc"].append(test_acc)
        
        if writer != None:
            writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss}, global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train_accuracy": train_acc, "test_accuracy": test_acc}, global_step=epoch)
            writer.add_graph(model=model, input_to_model=torch.randn(32, 3, 224, 224).to(device))

    if writer != None:
        writer.close()

    end_training_time = default_timer()
    print(f"[INFO] Training took {end_training_time-start_training_time:.3f} on {device.upper()}.")

    print(f"[INFO] Saving model to ./models/{model_save_name}.pth")
    MODEL_PATH = Path("./models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH / f"{model_save_name}.pth")
    print(f"[INFO] The model saved successfully.")
