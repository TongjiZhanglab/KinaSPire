import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .config import as_fit_params, load_config, save_config
from .data import DataSplitter, setup_dataloader_from_df, split_prior
from .evaluation import Evaluator
from .model import KinaSPire
from .pruning import prune_with_cpd
from ._utils import set_seeds


class Trainer(object):
    def __init__(self, num_epochs, output_path,
                 scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
                 scheduler_kwargs={'T_max': 10}, device='cpu'):
        super().__init__()
        self.num_epochs = num_epochs
        self.output_path = output_path
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs
        self.mse = nn.MSELoss()
        self.criterion = nn.MSELoss()
        self.device = device 

        os.makedirs(os.path.join(self.output_path, 'run_log'), exist_ok=True)

    def run_epochs(self, optimizer, model, train_loader, val_loader, cv, weight_decay, refit=None):
        logfile_path = os.path.join(self.output_path, 'run_log', f"cv{cv}_wd{weight_decay}_refit{refit}")
        os.makedirs(logfile_path, exist_ok=True)
        
        writer = SummaryWriter(logfile_path)
        logfile = open(os.path.join(logfile_path, 'log.txt'), 'a')
        scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
        
        train_epoch_loss = []
        valid_epoch_loss = []
        valid_epoch_r2 = []
        for epoch in range(self.num_epochs):
            train_loss_avg = self.train_iteration(model, train_loader, optimizer)
            val_loss_avg, val_r2_avg = self.valid_iteration(model, val_loader)

            scheduler.step()

            train_epoch_loss.append(train_loss_avg)
            valid_epoch_loss.append(val_loss_avg)
            valid_epoch_r2.append(val_r2_avg)
        
            # Write to tensorboard
            writer.add_scalar('Train/Loss', train_loss_avg, epoch)
            writer.add_scalar('Val/Loss', val_loss_avg, epoch)
            writer.add_scalar('Val/R2', val_r2_avg, epoch)

            # Write to logfile
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, Val R2: {val_r2_avg:.4f}", end='\r')
            logfile.write(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, Val R2: {val_r2_avg:.4f}\n")
        
        print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, Val R2: {val_r2_avg:.4f}\n")
            
        # Close SummaryWriter
        writer.close()

        # Close logfile
        logfile.flush()
        logfile.close()

        # Save report
        np.save(os.path.join(logfile_path, 'train_epoch_loss.npy'), train_epoch_loss)
        np.save(os.path.join(logfile_path, 'valid_epoch_loss.npy'), valid_epoch_loss)
        np.save(os.path.join(logfile_path, 'valid_epoch_r2.npy'), valid_epoch_r2)

        return {'train_loss': train_epoch_loss[-1], 
                'val_loss': valid_epoch_loss[-1], 
                'val_r2': valid_epoch_r2[-1],  ## the last epoch
                'activation': model.activation.__class__.__name__,
                'drop_p_prior': model.prior.dropout_rate,
                'drop_p_kki': model.kki.dropout_rate,
                'drop_p_ksr': model.ksr.dropout_rate
                }
               
    def train_iteration(self, model, train_loader, optimizer):
        model.train()
        
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            output = model(inputs)
            
            # loss = self.criterion(output, targets, self.alpha, self.threshold_right, self.threshold_left)
            loss = self.criterion(output, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss_avg = train_loss / len(train_loader)

        return train_loss_avg

    def valid_iteration(self, model, val_loader):
        val_loss, val_r2 = 0, 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            model.eval()
            with torch.no_grad():
                output = model(inputs)
            
            # loss = self.criterion(output, targets, self.alpha, self.threshold_right, self.threshold_left)
            loss_mse = self.mse(output, targets)
            loss = self.criterion(output, targets)

            var = torch.var(targets)
            r2 = 1 - (loss_mse.item() / var)
            
            val_loss += loss.item()
            val_r2 += r2.item()

        val_loss_avg = val_loss / len(val_loader)
        val_r2_avg = val_r2 / len(val_loader)

        return val_loss_avg, val_r2_avg


class OptimizerSetter(object):
    """
    :param optimizer_class: default: torch.optim.Adam
    :param optimizerkw: dict, key value pairs sent to the optimizer class, note that 
                        weight_decay and learning rate are supplied separately, default: {}
    :param optimizer_paramskw: dict, dictionary with keys corresponding to model attributes that 
                               should be tuned with the specific keywords specified by the value of 
                               that keys dictionary, for example, {'fc1': {'weight_decay': 1e-10}},
                               default: {}
    :param lr: learning rate, default: 1e-3
    :param weight_decay: L2 penalty supplied to the optimizer to stabilize parameter estimation, default: 0
    """
    def __init__(self, optimizer_class=torch.optim.Adam, optimizerkw={}, optimizer_paramskw={}, lr=1e-3, weight_decay=0, relmax=None, it=None):
        super().__init__()
        self.optimizer_class = optimizer_class
        self.optimizerkw = optimizerkw
        self.optimizer_paramskw = optimizer_paramskw
        self.lr = lr
        self.weight_decay = weight_decay
        self.relmax = relmax
        self.it = it

    def generate_optimizer(self, model):
        if not self.optimizer_paramskw:
            optimizer = self.optimizer_class(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, **self.optimizerkw)
        else:
            paramlist = []
            for k, v in model.named_parameters():
                key = k.split('.')[0]
                if key in self.optimizer_paramskw:
                    paramlist.append({'params': v, **self.optimizer_paramskw[key]})
                else:
                    paramlist.append({'params': v})

            optimizer = self.optimizer_class(paramlist, lr=self.lr, weight_decay=self.weight_decay, **self.optimizerkw)

        self.optimizer = optimizer

    def get_hyper_params(self):
        parameters = {}

        parameters['lr'] = self.lr
        parameters['weight_decay'] = self.weight_decay

        parameters['optimizer'] = self.optimizer_class.__name__
        for k, v in self.optimizerkw.items():
            parameters[k] = v

        for k, v in self.optimizer_paramskw.items():
            for kk, vv in v.items():
                parameters[f'{k}_{kk}'] = vv

        parameters['relmax'] = self.relmax
        parameters['it'] = self.it

        return parameters

def fit_model(data, prior, config=None, ppi_mask=None, output_path='./test_result'):
    """
    Main training routine.

    Hyperparameters are read from a YAML file or dict. ``data``, ``prior``,
    ``ppi_mask`` and ``output_path`` remain runtime arguments.

    :param config: YAML path, nested dict, or None to use package defaults
    """
    cfg = load_config(config)
    p = as_fit_params(cfg)

    os.makedirs(output_path, exist_ok=True)
    save_config(cfg, os.path.join(output_path, 'config.yaml'))

    data_val_size = p.data_val_size
    batch_size = p.batch_size
    fraction_gs = p.fraction_gs
    num_epochs = p.num_epochs
    cvs = p.cvs
    num_epochs_refit = p.num_epochs_refit
    refit_iters = p.refit_iters
    refit_resample = p.refit_resample
    weight_decays = p.weight_decays
    lr = p.lr
    scheduler_class = p.scheduler_class
    scheduler_kwargs = p.scheduler_kwargs
    optimizer_class = p.optimizer_class
    optimizerkw = p.optimizerkw
    optimizer_paramskw = p.optimizer_paramskw
    dropout_rate1 = p.dropout_rate1
    dropout_rate2 = p.dropout_rate2
    dropout_rate3 = p.dropout_rate3
    eps = p.eps
    eps_factor = p.eps_factor
    fill_zeroed = p.fill_zeroed
    device = p.device

    # Global container
    data_splitter = DataSplitter(data=data, data_val_size=data_val_size, data_test_size=0)
    trainer = Trainer(num_epochs=num_epochs, output_path=output_path,
                    scheduler_class=scheduler_class,
                    scheduler_kwargs=scheduler_kwargs,
                    device=device)
    evaluator = Evaluator(output_path=output_path)

    # Create output dir
    if not os.path.exists(os.path.join(output_path, 'model')):
        os.makedirs(os.path.join(output_path, 'model'), exist_ok=True)

    for cv in range(cvs):
        # 1-1. Split samples and setup dataloader
        train_data, val_data, _ = data_splitter.split(random_state=cv)
        train_loader = setup_dataloader_from_df(df=train_data, batch_size=batch_size, shuffle=True)
        val_loader = setup_dataloader_from_df(df=val_data, batch_size=batch_size, shuffle=False)
        
        # 1-2. Split prior
        rng = np.random.default_rng(3238 + cv)
        train_prior, test_prior = split_prior(prior=prior, rng=rng, fraction_gs=fraction_gs)
        
        # num_total_ksr = (prior > 0).sum().sum()
        # num_train_ksr = (train_prior > 0).sum().sum()
        # num_test_ksr = (test_prior > 0).sum().sum()
        # print(f"Total number of KSRs: {num_total_ksr}, number of KSRs used to train: {num_train_ksr}, number of KSRs used to test: {num_test_ksr}\n")

        if isinstance(weight_decays, tuple):
            weight_decays = np.logspace(*weight_decays)
        
        for wd in weight_decays:
            print(f"Training progress: cv {cv + 1}/{cvs} | weight_decay={wd} | fit")

            # 2-1. Model initialization
            set_seeds(cv)
            model = KinaSPire(input_dim=train_data.shape[1], hidden_dim=train_prior.shape[0], 
                            ksr_mask=train_prior, ppi_mask=ppi_mask, use_ksr_as_weights=False,
                            dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2, 
                            dropout_rate3=dropout_rate3).to(device)
            
            # 2-2. Optimizer initialization
            opt_setter = OptimizerSetter(optimizer_class=optimizer_class, optimizerkw=optimizerkw, 
                                        optimizer_paramskw=optimizer_paramskw, lr=lr, weight_decay=wd, relmax=None, it=None)
            opt_setter.generate_optimizer(model)
    
            # 3. Fit model
            fit_report = trainer.run_epochs(optimizer=opt_setter.optimizer, model=model, train_loader=train_loader, 
                                            val_loader=val_loader, cv=cv, weight_decay=wd, refit=None)
            
            # 4. Record training performance
            evaluator.run_evaluation(cv=cv, weight_decay=wd, refit=None, train_report=fit_report)
            evaluator.save_performance(save_tmp=True)
    
            # 5. Save model
            model_save_path = os.path.join(output_path, 'model', f"cv{cv}_wd{wd}_refitNone.pth")
            torch.save(model.cpu(), model_save_path)
            model = model.to(device)
    
            # 6. Refit
            refit_model(model=model, data_splitter=data_splitter, trainer=trainer, evaluator=evaluator, 
                        train_loader=train_loader, val_loader=val_loader, prior=train_prior, 
                        batch_size=batch_size, num_epochs=num_epochs_refit, refit_iters=refit_iters, 
                        resample=refit_resample, eps=eps, eps_factor=eps_factor, fill_zeroed=fill_zeroed, 
                        optimizer_class=optimizer_class, optimizerkw=optimizerkw, optimizer_paramskw=optimizer_paramskw, 
                        lr=lr, wd=wd, relmax=0, cv=cv, cvs=cvs, output_path=output_path, device=device)
    
    # Store final performance
    evaluator.save_performance(save_tmp=False)


def refit_model(model, data_splitter, trainer, evaluator, train_loader, val_loader, 
                prior, batch_size=64, num_epochs=30, refit_iters=5, resample=True, 
                eps=torch.finfo(torch.float).eps, eps_factor=10, fill_zeroed=True, 
                optimizer_class=torch.optim.Adam, optimizerkw={}, 
                optimizer_paramskw={'prior': {'weight_decay': 1e-10}}, lr=1e-4, wd=0,
                relmax=0, cv=0, cvs=1, output_path='./test_result', device='cpu'):
    """
    Prune weights of kki and ksr based on CPD and refit model.
    """
    init_epochs = trainer.num_epochs
    trainer.num_epochs = num_epochs
    
    for refit in range(refit_iters):
        print(f"Training progress: cv {cv + 1}/{cvs} | weight_decay={wd} | refit {refit + 1}/{refit_iters}")

        # 1. Resample
        if resample:
            train_data, val_data, _ = data_splitter.split(random_state=refit + 2024)
            train_loader = setup_dataloader_from_df(df=train_data, batch_size=batch_size, shuffle=True)
            val_loader = setup_dataloader_from_df(df=val_data, batch_size=batch_size, shuffle=False)
    
        # 2. Prune based on CPD
        prune_with_cpd(model=model, data_loader=val_loader, eps=eps, eps_factor=eps_factor, fill_zeroed=fill_zeroed, device=device)
            
        # 3. Optimizer initialization
        opt_setter = OptimizerSetter(optimizer_class=optimizer_class, optimizerkw=optimizerkw, 
                                     optimizer_paramskw=optimizer_paramskw, lr=lr, 
                                     weight_decay=wd, relmax=relmax, it=refit)
        opt_setter.generate_optimizer(model)
        # print("refit", refit, opt_setter.optimizer.param_groups[0]['lr'])
    
        # 3. Refit model
        refit_report = trainer.run_epochs(optimizer=opt_setter.optimizer, model=model, train_loader=train_loader, 
                                          val_loader=val_loader, cv=cv, weight_decay=wd, refit=refit)
        
        # 4. Record training performance
        evaluator.run_evaluation(cv=cv, weight_decay=wd, refit=refit, train_report=refit_report)
        evaluator.save_performance(save_tmp=True)
    
        # 5. Save model
        model_save_path = os.path.join(output_path, 'model', f"cv{cv}_wd{wd}_refit{refit}.pth")
        torch.save(model.cpu(), model_save_path)
        model = model.to(device)

    trainer.num_epochs = init_epochs

