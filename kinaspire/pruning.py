import copy
import torch
import torch.nn.utils.prune as prune


def cal_cpd(model, data_loader, node_indices, device='cpu'):
    """
    Calculate coefficient of partial determination (CPD) when delete a specific kinase.

    :param model: trained model
    :param data_loader: phos-MS data loader
    :param node_indices: kinase indices to be deleted
    """
    # Save a copy of the original model
    original_model = copy.deepcopy(model)
        
    # Calculate original output
    original_mse = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        model.eval()
        with torch.no_grad():
            original_output = model(inputs)
    
        mse = ((original_output - targets) ** 2).mean(dim=0)
        original_mse += mse
    original_mse = (original_mse / len(data_loader)).detach()

    # Calculate perturbed output
    ## KO kinase
    perturbed_mse = []
    for node_index in node_indices:
        single_perturbed_mse = cal_perturbed_mse(original_model, data_loader, node_index, layer=2, device=device)
        perturbed_mse.append(single_perturbed_mse)
    
    perturbed_mse = torch.stack(perturbed_mse, 0)
    # print(perturbed_mse.size())
    
    kin_cpd = 1 - original_mse / perturbed_mse

    ## KO meta kinase
    perturbed_mse = []
    for node_index in node_indices:
        single_perturbed_mse = cal_perturbed_mse(original_model, data_loader, node_index, layer=3, device=device)
        perturbed_mse.append(single_perturbed_mse)
    
    perturbed_mse = torch.stack(perturbed_mse, 0)    
    mkin_cpd = 1 - original_mse / perturbed_mse

    return kin_cpd, mkin_cpd


def cal_perturbed_mse(original_model, data_loader, node_index, layer=2, device='cpu'):
    model_copy = copy.deepcopy(original_model)
    if layer==2:
        model_copy.set_mask1(node_index)
        print(f"Delete kinase {model_copy.kinase_mask1} in layer 2.", end='\r')
    else:
        model_copy.set_mask2(node_index)
        print(f"Delete meta kinase {model_copy.kinase_mask2} in layer 3.", end='\r')

    perturbed_mse = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        model_copy.eval()
        with torch.no_grad():
            perturbed_output = model_copy(inputs)
    
        mse = ((perturbed_output - targets) ** 2).mean(dim=0)
        perturbed_mse += mse
    perturbed_mse = (perturbed_mse / len(data_loader)).detach()

    return perturbed_mse


def threshold_cpd(cpd, eps):
    cpd[cpd.abs() < eps] = 0
    cpd_mask = (cpd > 0).int()
    cpd = cpd * cpd_mask
    cpd[cpd < eps] = 0

    return cpd, cpd_mask


def cal_kk_mean_cpd(kin_cpd, kin_mask, mkin_cpd, mkin_mask):
    """
    Note that all the input shapes are site * kinase.
    """
    M1 = mkin_cpd * mkin_mask
    M2 = kin_cpd * kin_mask
    
    kk_nnz_joint = mkin_mask.T.float() @ kin_mask.float()
    kk_nnz_joint = torch.nan_to_num(kk_nnz_joint, nan=0.0, neginf=0.0).int()

    kk_mean_cpd = (M1.T @ M2) / kk_nnz_joint
    kk_mean_cpd = torch.nan_to_num(kk_mean_cpd, nan=0.0, neginf=0.0)
    
    return kk_mean_cpd


def get_old_mask(module, masked=True, device='cpu'):
    if hasattr(module.fc, 'weight_mask') and masked:
        old_mask = (module.fc.weight_mask.detach() > 0).to(torch.long)
    else:
        old_mask = torch.ones(module.fc.weight.size(), device=device)

    return old_mask


def prune_step(module, new_mask, fill_zeroed=True):
    if prune.is_pruned(module.fc):
        # Removing previous mask
        prune.remove(module.fc, 'weight')

    indx = (new_mask.sum(1) == 0)
    if fill_zeroed and (indx.sum() > 0):
        new_mask[indx, :] = 1

    prune.custom_from_mask(module.fc, 'weight', new_mask)


def prune_with_cpd(model, data_loader, eps, eps_factor=10, fill_zeroed=True, device='cpu'):
    node_indices = list(range(model.hidden_dim))
    kin_cpd, mkin_cpd = cal_cpd(model, data_loader, node_indices, device=device)  ## kinase * site

    kin_cpd, kin_mask = threshold_cpd(kin_cpd, eps)
    kin_old_mask = get_old_mask(model.ksr, masked=False, device=device)  ## only for getting a matrix of 1s, site * kinase
    kin_new_mask = torch.logical_and(kin_mask.T, kin_old_mask).int()  ## site * kinase
    
    mkin_cpd, mkin_mask = threshold_cpd(mkin_cpd, eps)
    mkin_old_mask = get_old_mask(model.ksr, masked=True, device=device)  ## site * kinase
    mkin_new_mask = torch.logical_and(mkin_mask.T, mkin_old_mask).int()  ## site * kinase

    kk_cpd = cal_kk_mean_cpd(kin_cpd.T, kin_mask.T, mkin_cpd.T, mkin_mask.T)
    _, kk_mask = threshold_cpd(kk_cpd, eps / eps_factor)
    kk_old_mask = get_old_mask(model.kki, masked=True, device=device)
    kk_new_mask = torch.logical_and(kk_mask, kk_old_mask).int()

    prune_step(model.ksr, mkin_new_mask, fill_zeroed=fill_zeroed)
    prune_step(model.kki, kk_new_mask, fill_zeroed=False)

