import torch
import helpers
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Set a seed value
# seed = 42 

# # Set the seed for generating random numbers in PyTorch
# torch.manual_seed(seed)

# # Set the seed for NumPy (used in various data preprocessing steps)
# np.random.seed(seed)


# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

def unlearning_finetuning(model, retain, epochs):
    
    epochs = epochs

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()

    for _ in range(epochs):
        for inputs, targets in retain:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    return model

def unlearning_EWCU(model, retain, forget, epochs, threshold=0.00001):
    epochs = epochs

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()

    #print('Computing EFIM')
    efim = helpers.EFIM(model, forget)
    #print('EFIM Computed')
    
    parameters_to_freeze, _ = helpers.get_parameters_with_small_norm(efim, threshold)
    parameters_to_freeze = [param for param in parameters_to_freeze if param not in ['fc.weight', 'fc.bias']]

    helpers.freeze_parameters(model, parameters_to_freeze)


    for _ in range(epochs):
        for inputs, targets in retain:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    return model

def unlearning_EWCU_2(model, retain, forget, epochs, threshold=0.031):
    epochs = epochs

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()

    #print('Computing EFIM')
    efim = helpers.EFIM(model, forget)
    #print('EFIM Computed')
    agg_efim = helpers.aggregatedEFIM(efim)
    
    parameters_to_freeze = helpers.params_below_threshold(agg_efim, threshold)
    parameters_to_freeze = [param for param in parameters_to_freeze if param not in ['fc.weight', 'fc.bias']]

    helpers.freeze_parameters(model, parameters_to_freeze)


    for _ in range(epochs):
        for inputs, targets in retain:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    return model

def fisher_scrub(model, retain_loader, forget_loader, validation, epochs=5, threshold = 0.01):
    model_t = copy.deepcopy(model)
    model_s = model

    efim = helpers.EFIM(model_s, forget_loader)
    agg_efim = helpers.aggregatedEFIM(efim)

    parameters_to_freeze = helpers.params_below_threshold(agg_efim, threshold)
    parameters_to_freeze = [param for param in parameters_to_freeze if param not in ['fc.weight', 'fc.bias']]
    helpers.freeze_parameters(model_s, parameters_to_freeze)

    swa_model = torch.optim.swa_utils.AveragedModel(model_s, avg_fn=helpers.avg_fn)

    alpha = 0.001
    gamma = 0.99
    beta = 0
    smoothing = 0
    msteps = epochs
    clip = 0.2
    sstart = 4
    kd_T = 4
    distill = 'kd'

    sgda_epochs = epochs # This is larger than what is set for the competition
    sgda_learning_rate = 0.0005
    sgda_learning_rate_min = 0.00005

    lr_decay_epochs = [7,8,9]
    lr_decay_rate = 0.1
    sgda_weight_decay = 5e-4
    sgda_momentum = 0.9

    # Modules
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    # Losses
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = helpers.DistillKL(kd_T)
    criterion_kd = helpers.DistillKL(kd_T)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    optimizer = optim.SGD(trainable_list.parameters(),
                                  lr=sgda_learning_rate,
                                  momentum=sgda_momentum,
                                  weight_decay=sgda_weight_decay)

    #Last is always good teacher
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True
        swa_model.cuda()

    for epoch in range(1, sgda_epochs + 1):
        lr = helpers.adjust_learning_rate(epoch, lr_decay_rate, sgda_learning_rate , lr_decay_epochs, optimizer)

        maximize_loss = 0

        if epoch <= msteps:
            lr = helpers.adjust_learning_rate(epoch, lr_decay_rate, sgda_learning_rate , lr_decay_epochs, optimizer)
            maximize_loss = helpers.train_distill(epoch, forget_loader, module_list, swa_model, criterion_list, optimizer, distill, gamma, alpha, beta, smoothing, "maximize")
            

        lr = helpers.adjust_learning_rate(epoch, lr_decay_rate, sgda_learning_rate_min , lr_decay_epochs, optimizer)
        train_loss = helpers.train_distill(epoch, retain_loader, module_list, swa_model, criterion_list, optimizer, distill, gamma, alpha, beta, smoothing, "minimize")
        

        if epoch >= sstart:
            swa_model.update_parameters(model_s)

    model_s.eval()
    return model_s


def unlearning_ts(model, retain_loader, forget_loader, validation, epochs=5):
    #Create the teacher and the student
    model_t = copy.deepcopy(model)
    model_s = model

    swa_model = torch.optim.swa_utils.AveragedModel(model_s, avg_fn=helpers.avg_fn)

    alpha = 0.001
    gamma = 0.99
    beta = 0
    smoothing = 0
    msteps = epochs
    clip = 0.2
    sstart = 4
    kd_T = 4
    distill = 'kd'

    sgda_epochs = epochs
    sgda_learning_rate = 0.0005
    sgda_learning_rate_min = 0.00005

    lr_decay_epochs = [7,8,9]
    lr_decay_rate = 0.1
    sgda_weight_decay = 5e-4
    sgda_momentum = 0.9

    # Modules
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    # Losses
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = helpers.DistillKL(kd_T)
    criterion_kd = helpers.DistillKL(kd_T)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    optimizer = optim.SGD(trainable_list.parameters(),
                                  lr=sgda_learning_rate,
                                  momentum=sgda_momentum,
                                  weight_decay=sgda_weight_decay)

    #Last is always good teacher
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True
        swa_model.cuda()

    for epoch in range(1, sgda_epochs + 1):
        lr = helpers.adjust_learning_rate(epoch, lr_decay_rate, sgda_learning_rate , lr_decay_epochs, optimizer)

        maximize_loss = 0

        if epoch <= msteps:
            lr = helpers.adjust_learning_rate(epoch, lr_decay_rate, sgda_learning_rate , lr_decay_epochs, optimizer)
            maximize_loss = helpers.train_distill(epoch, forget_loader, module_list, swa_model, criterion_list, optimizer, distill, gamma, alpha, beta, smoothing, "maximize")
            

        lr = helpers.adjust_learning_rate(epoch, lr_decay_rate, sgda_learning_rate_min , lr_decay_epochs, optimizer)
        train_loss = helpers.train_distill(epoch, retain_loader, module_list, swa_model, criterion_list, optimizer, distill, gamma, alpha, beta, smoothing, "minimize")
        

        if epoch >= sstart:
            swa_model.update_parameters(model_s)

    model_s.eval()
    return model_s


def blindspot_unlearner(model, unlearning_teacher, full_trained_teacher, combined_loader, epochs = 5,
                optimizer = 'adam', lr = 0.01, 
                device = 'cuda', KL_temperature = 1):
    # creating the unlearning dataset.
    unlearning_loader = combined_loader

    unlearning_teacher.eval()
    full_trained_teacher.eval()

    optimizer = optimizer
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):
        loss = helpers.unlearning_step(model = model, unlearning_teacher= unlearning_teacher, 
                        full_trained_teacher=full_trained_teacher, unlearn_data_loader=unlearning_loader, 
                        optimizer=optimizer, device=device, KL_temperature=KL_temperature)
        #print("Epoch {} Unlearning Loss {}".format(epoch+1, loss))

    return model

def cf_k(model, retain_loader, epochs=5):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Freeze all the parameters.
    for param in model.parameters():
        param.requires_grad = False

    # Only unfreeze the last three layers for the fine-tuning.
    # for param in model.layer2.parameters():
    #     param.requires_grad = True
    # for param in model.layer3.parameters():
    #     param.requires_grad = True
    # for param in model.layer4.parameters():
    #     param.requires_grad = True
    for param in model.avgpool.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    num_epochs = epochs


    for epoch in range(num_epochs):
        running_loss = 0

        for batch_idx, (x_retain, y_retain) in enumerate(retain_loader):
            y_retain = y_retain.cuda()

            # Classification Loss
            outputs_retain = model(x_retain.cuda())
            classification_loss = criterion(outputs_retain, y_retain)

            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()

            running_loss += classification_loss.item() * x_retain.size(0)
            # print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(retain_loader)}] - Batch Loss: {classification_loss.item():.4f}")
        #scheduler.step() #I added this
    #     average_epoch_loss = running_loss / (len(retain_loader) * x_retain.size(0))
    #     print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}")

    return model


def neg_grad(model, retain_loader, forget_loader, epochs=5):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    dataloader_iterator = iter(forget_loader)

    num_epochs = epochs
    for epoch in range(num_epochs):
        running_loss = 0

        for batch_idx, (x_retain, y_retain) in enumerate(retain_loader):
            y_retain = y_retain.cuda()

            try:
                (x_forget, y_forget) = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(forget_loader)
                (x_forget, y_forget) = next(dataloader_iterator)

            if x_forget.size(0) != x_retain.size(0):
                continue

            outputs_forget = model(x_forget.cuda())
            loss_ascent_forget = -criterion(outputs_forget, y_forget.cuda())

            optimizer.zero_grad()
            loss_ascent_forget.backward()
            optimizer.step()

            running_loss += loss_ascent_forget.item() * x_retain.size(0)
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(retain_loader)}] - Batch Loss: {loss_ascent_forget.item():.4f}")

        average_epoch_loss = running_loss / (len(retain_loader) * x_retain.size(0))
        print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}")

    return model

def advanced_neg_grad(model, retain_loader, forget_loader, epochs=5):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    dataloader_iterator = iter(forget_loader)

    num_epochs = epochs
    for epoch in range(num_epochs):
        running_loss = 0

        for batch_idx, (x_retain, y_retain) in enumerate(retain_loader):
            y_retain = y_retain.cuda()

            try:
                (x_forget, y_forget) = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(forget_loader)
                (x_forget, y_forget) = next(dataloader_iterator)

            if x_forget.size(0) != x_retain.size(0):
                continue

            outputs_retain = model(x_retain.cuda())
            outputs_forget = model(x_forget.cuda())

            loss_ascent_forget = -criterion(outputs_forget, y_forget.cuda())
            loss_retain = criterion(outputs_retain, y_retain.cuda())

            # Overall loss
            joint_loss = loss_ascent_forget + loss_retain

            print("joint loss :", joint_loss.item())
            optimizer.zero_grad()
            joint_loss.backward()
            optimizer.step()

            running_loss += joint_loss.item() * x_retain.size(0)
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(retain_loader)}] - Batch Loss: {joint_loss.item():.4f}")

        average_epoch_loss = running_loss / (len(retain_loader) * x_retain.size(0))
        print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}")

    return model


def unsir(model, retain_loader, forget_loader, epochs=5):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    print_interval = 1
    train_epoch_losses = []

    num_epochs = epochs
    for epoch in range(num_epochs):
        running_loss = 0

        for batch_idx, ((x_retain, y_retain), (x_forget, y_forget)) in enumerate(zip(retain_loader, forget_loader)):
            y_retain = y_retain.cuda()
            batch_size_forget = y_forget.size(0)

            if x_retain.size(0) != 64 or x_forget.size(0) != 64:
                continue

            # Initialize the noise.
            noise_dim = x_retain.size(1), x_retain.size(2), x_retain.size(3)
            noise = helpers.Noise(batch_size_forget, *noise_dim).cuda()
            noise_optimizer = torch.optim.Adam(noise.parameters(), lr=0.01)
            noise_tensor = noise()[:batch_size_forget]

            # Update the noise for increasing the loss value.
            for _ in range(5):
                outputs = model(noise_tensor)
                with torch.no_grad():
                    target_logits = model(x_forget.cuda())
                # Maximize the similarity between noise data and forget features.
                loss_noise = -F.mse_loss(outputs, target_logits)

                # Backpropagate to update the noise.
                noise_optimizer.zero_grad()
                loss_noise.backward(retain_graph=True)
                noise_optimizer.step()

            # Train the model with noise and retain image
            noise_tensor = torch.clamp(noise_tensor, 0, 1).detach().cuda()
            outputs = ModuleNotFoundError(noise_tensor.cuda())
            loss_1 = criterion(outputs, y_retain)

            outputs = model(x_retain.cuda())
            loss_2 = criterion(outputs, y_retain)

            joint_loss = loss_1 + loss_2

            optimizer.zero_grad()
            joint_loss.backward()
            optimizer.step()
            running_loss += joint_loss.item() * x_retain.size(0)

            
            original_image = x_retain[0].cpu().numpy().transpose(1, 2, 0)
            image1 = TF.to_pil_image(float_to_uint8(original_image))
            image2 = TF.to_pil_image(noise.noise[0].cpu())

            # Display original image.
            original_image = x_retain[0].cpu().numpy().transpose(1, 2, 0)
            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)  # 2 rows, 2 columns, position 1
            plt.imshow(image1)
            plt.title("Original Image")
            plt.axis('off')

            # Display first noise image.
            plt.subplot(1, 2, 2)
            plt.imshow(image2)
            plt.title("Noise Image")
            plt.axis('off')

            # Show all the subplots.
            plt.tight_layout()
            plt.show()


            if batch_idx % print_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(retain_loader)}] - Batch Loss: {joint_loss.item():.4f}")

        average_train_loss = running_loss / (len(retain_loader) * x_retain.size(0))
        train_epoch_losses.append(average_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {average_train_loss:.4f}")
    
    print('------------Finished Unsir Stage 1--------------------')

    # num_epochs = 1
    # for epoch in range(num_epochs):
    #     running_loss = 0

    #     for batch_idx, (x_retain, y_retain) in enumerate(retain_loader):
    #         y_retain = y_retain.cuda()

    #         # Classification Loss
    #         outputs_retain = model(x_retain.cuda())
    #         classification_loss = criterion(outputs_retain, y_retain)

    #         optimizer.zero_grad()
    #         classification_loss.backward()
    #         optimizer.step()

    #         running_loss += classification_loss.item() * x_retain.size(0)
    #         print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(retain_loader)}] - Batch Loss: {classification_loss.item():.4f}")

    #     average_epoch_loss = running_loss / (len(retain_loader) * x_retain.size(0))
    #     print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}")

    return model


def ZS_EWCU(model, forget, threshold=1e-6, mode='noise'):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    model.train()

    
    efim = helpers.EFIM(model, forget)

    count_above_threshold = sum(torch.sum(param > threshold).item() for param in efim.values())
    print(f"Number of weights above threshold: {count_above_threshold}")
    
    if mode=='zero':
        counter = 0
        
        for (name, efim_param), (model_name, model_param) in zip(efim.items(), model.named_parameters()):
            mask = efim_param > threshold
            zeros_before = torch.sum(model_param.data == 0.0)
            model_param.data[mask] = 0.0
            zeros_after = torch.sum(model_param.data == 0.0)
            counter += (zeros_after - zeros_before).item()

        print(f"Total parameters set to zero: {counter}")

    elif mode=='noise':
        noise_added_count = 0
        noise_reduction_factor = 0.1
        for (name, efim_param), (model_name, model_param) in zip(efim.items(), model.named_parameters()):
            mask = efim_param > threshold
            # Use the absolute value of the parameters as the scale for the standard deviation
            scaled_std = torch.abs(model_param.data) * noise_reduction_factor
            noise = torch.randn_like(model_param.data) * scaled_std
            model_param.data[mask] += noise[mask]
            noise_added_count += mask.sum().item()

        print(f"Number of parameters with noise added: {noise_added_count}")

    return model


