import torch
from torch import nn
import numpy as np
from sklearn import linear_model, model_selection
import torch.nn.functional as F



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_losses(net, loader):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def eval_mia(model, train_loader, test_loader, forget_loader):

  #Loader were defined previosly

  train_losses = compute_losses(model, train_loader)
  test_losses = compute_losses(model, test_loader)
  forget_losses = compute_losses(model, forget_loader)

  # Since we have more forget losses than test losses, sub-sample them, to have a class-balanced dataset.
  np.random.shuffle(forget_losses)
  forget_losses = forget_losses[: len(test_losses)]

  samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
  labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)

  mia_scores = simple_mia(samples_mia, labels_mia)

  print(
      f"The MIA has an accuracy of {mia_scores.mean():.3f} on forgotten vs unseen images"
  )
  return mia_scores.mean()


def accuracy(model, loader):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


# Adding KL-divergence as a new metric for unlearning
def compute_kl_divergence(model1, model2, dataloader):
    model1.eval()
    model2.eval()
    total_kl_div = 0.0

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(DEVICE)
            log_probs1 = F.log_softmax(model1(inputs), dim=1)
            probs2 = F.softmax(model2(inputs), dim=1)
            kl_div = F.kl_div(log_probs1, probs2, reduction='batchmean')
            total_kl_div += kl_div.item() * inputs.size(0)

    return total_kl_div / len(dataloader.dataset)