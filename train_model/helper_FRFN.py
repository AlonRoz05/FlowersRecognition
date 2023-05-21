import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import random

def show_model_quality(model: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader, test_dataset, class_names, device):
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(test_dataset), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in test_samples:
            sample = torch.unsqueeze(sample, dim=0).to(device)

            pred_logits = model(sample)
            pred_prob = torch.softmax(pred_logits, dim=1)
            pred_probs.append(pred_prob.cpu())

    pred_probs = torch.stack(pred_probs)
    pred_classes = pred_probs.max(dim=1).indices

    plt.figure(figsize=(9,9))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(torch.squeeze(sample).permute(1,2,0))
        plt.axis(False)
        
        print(pred_classes[i])
        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]]

        title_text = f"Pred: {pred_label} | Truth: {truth_label} | Pred probability: {torch.max(pred_prob):.3f}"
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")
        else:
            plt.title(title_text, fontsize=10, c="r")

    plt.show()

    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Making predictions..."):
            X, y = X.to(device), y.to(device)
            y_logit = model(X)
            y_pred = torch.softmax(y_logit.squeeze(), dim=0).max(dim=1).indices
            y_preds.append(y_pred.cpu())

    y_pred_tensor = torch.cat(y_preds)
    confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
    confmat_tensor = confmat(preds=y_pred_tensor, target=test_dataset.targets)

    fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(), class_names=class_names, figsize=(10,7))
    plt.show()
