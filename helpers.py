import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def EFIM(model, data_loader):
    # Initialize EFIM dictionary with zeros for each learnable parameter
    efim = {name: torch.zeros_like(param, device=param.device)
            for name, param in model.named_parameters() if param.requires_grad}

    model.to(device)
    model.train()

    # Accumulate the squared gradients over all batches
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        model.zero_grad()
        outputs = model(inputs)
        loss = cross_entropy(outputs, targets)
        loss.backward()


        for name, param in model.named_parameters():
            if param.grad is not None:
                efim[name] += param.grad.data ** 2
                #efim[name] += param.grad

    # Average over the number of data points
    num_data_points = len(data_loader.dataset)
    for name in efim:
        efim[name] /= num_data_points

    return efim

def get_parameters_with_small_norm(efim, threshold):
    """Get parameters with a norm smaller than the threshold."""
    small_norm_params = []
    large_norm_params = []
    for name, value_tensor in efim.items():
        if value_tensor.norm() < threshold:
            small_norm_params.append(name)
        else:
          large_norm_params.append(name)
    return small_norm_params, large_norm_params

def freeze_parameters(model, list_parameters):
  for name, param in model.named_parameters():
    if name in list_parameters:
      param.requires_grad = False

def count_frozen_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)

def params_below_threshold(aggregatedEFIM, threshold):
    params = []
    for param_name, value in aggregatedEFIM:
       if value < threshold:
          params.append(param_name)
    return params
          

def aggregatedEFIM(efim):
   aggregated_efim = {name: efim[name].sum().item() for name in efim}
   aggregated_efim = sorted(aggregated_efim.items(), key=lambda x: x[1], reverse=True)
   return aggregated_efim

#this is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
#Specific for mix-max problems

def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
  beta = 0.1
  return (1 - beta) * averaged_model_parameter + beta * model_parameter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
    

def adjust_learning_rate(epoch, lr_decay_rate, sgda_learning_rate , lr_decay_epochs , optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = sgda_learning_rate
    if steps > 0:
        new_lr = sgda_learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr


def param_dist(model, swa_model, p):
    #This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist

def train_distill(epoch, train_loader, module_list, swa_model, criterion_list, optimizer, distill, gamma, alpha, beta, smoothing, split, quiet=True, bt=False, rf_t=False):
    """One epoch distillation"""

    #print('train_distill')

    # set modules as train()
    for module in module_list:
        module.train()

    if rf_t:
      module_list[1].eval()


    # set teacher as eval()
    module_list[-1].eval()



    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    if rf_t:
      print('getting rft')
      model_rf_t= module_list[1]



    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()

    for idx, data in enumerate(train_loader):
        if distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target = data

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if distill in ['crd']:
                contrast_idx = contrast_idx.cuda()
                index = index.cuda()

        # ===================forward=====================

        logit_s = model_s(input)



        with torch.no_grad():
            logit_t = model_t(input)

        if rf_t and split == "maximize":
          logit_t = model_rf_t(input)


        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)


        # other kd beyond KL divergence
        if distill == 'kd':
            loss_kd = 0

        else:
            raise NotImplementedError(distill)

        # Here I eliminated the other losses

        if split == "minimize":
            loss = gamma * loss_cls + alpha * loss_div + beta * loss_kd

        elif split == "maximize":
            loss = -loss_div

        loss = loss + param_dist(model_s, swa_model, smoothing)

        if split == "minimize":

            losses.update(loss.item(), input.size(0))


        elif split == "maximize":
            kd_losses.update(loss.item(), input.size(0))


        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


    if split == "minimize":
        return losses.avg
    else:
        return kd_losses.avg

def UnlearnerLoss_Bad_T(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim = 1)
    
    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)

def unlearning_step(model, unlearning_teacher, full_trained_teacher, unlearn_data_loader, optimizer, 
            device, KL_temperature):
    losses = []
    for batch in unlearn_data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss_Bad_T(output = output, labels=y, full_teacher_logits=full_teacher_logits, 
                unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)



def combine_loaders(forget_loader, retain_loader):

    combined_data = []
    combined_labels = []

    # Process forget_dataloader and assign label 1
    for data, _ in forget_loader:
        combined_data.append(data)
        combined_labels.append(torch.ones(len(data)))  # Create a tensor of ones

    # Process retain_dataloader and assign label 0
    for data, _ in retain_loader:
        combined_data.append(data)
        combined_labels.append(torch.zeros(len(data)))  # Create a tensor of zeros

    # Concatenate 
    combined_data = torch.cat(combined_data, dim=0)
    combined_labels = torch.cat(combined_labels, dim=0)

    # Create a new dataset from the combined data and labels
    combined_dataset = TensorDataset(combined_data, combined_labels)

    # Create a new dataloader from the combined dataset
    new_dataloader = DataLoader(combined_dataset, batch_size=256, shuffle=True)
    return new_dataloader

class Noise(nn.Module):
    def __init__(self, batch_size, *dim):
        super().__init__()
        self.noise = nn.Parameter(torch.randn(batch_size, *dim), requires_grad=True)

    def forward(self):
        return self.noise

def float_to_uint8(img_float):
    """Convert a floating point image in the range [0,1] to uint8 image in the range [0,255]."""
    img_uint8 = (img_float * 255).astype(np.uint8)
    return img_uint8


