import glob
import os
from collections import defaultdict
from pathlib import Path

import math
import numpy as np
import torch
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from utils.logger_utils import create_url_shortcut_of_wandb, create_logger_of_wandb
from utils.train_utils import SmoothedValue, set_random_seed
from utils.import_utils import fill_args_from_dict
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model.train_val_forward import simple_train_val_forward
from omegaconf import OmegaConf


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def exists(x):
    return x is not None


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def composite_onehot_to_mask(onehot_pred):
    pred = onehot_pred[0] if onehot_pred.dim() == 4 else onehot_pred
    composite_array = torch.zeros((pred.shape[1], pred.shape[2]), dtype=torch.uint8)
    
    # Apply argmax to get the most likely class for each pixel
    pred_classes = torch.argmax(pred, dim=0)
    
    # Create composite with better contrast
    for i in range(pred.shape[0]):
        mask = (pred_classes == i)
        composite_array[mask] = i * (255 // (pred.shape[0] - 1)) if pred.shape[0] > 1 else 255
    
    return composite_array

def cal_mae(gt, res, thresholding, save_to=None, n=None):
    res = np.array(res).squeeze()
    gt = np.array(gt).squeeze()

    mae_per_channel = []
    for c in range(gt.shape[0]):
        channel_mae = np.sum(np.abs(res[c] - gt[c])) * 1.0 / (gt.shape[1] * gt.shape[2])
        mae_per_channel.append(channel_mae)
    mae = np.mean(mae_per_channel)
    
    if save_to is not None:
        composite = np.zeros(gt.shape[1:], dtype=np.float32)
        for c in range(res.shape[0]):
            composite[res[c] > 0.5] = c * 50
        plt.imsave(os.path.join(save_to, n), composite, cmap='gray')
    
    return mae


def run_on_seed(func):
    def wrapper(*args, **kwargs):
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        set_random_seed(0)
        res = func(*args, **kwargs)
        set_random_seed(seed)
        return res

    return wrapper


class Trainer(object):
    def __init__(
            self,
            model,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader = None,
            train_val_forward_fn=simple_train_val_forward,
            gradient_accumulate_every=1,
            optimizer=None, scheduler=None,
            train_num_epoch=100,
            results_folder='./results',
            amp=False,
            fp16=False,
            split_batches=True,
            log_with='wandb',
            cfg=None,
    ):
        super().__init__()
        """
            Initialize the accelerator.
        """
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            log_with='wandb' if log_with else None,
            gradient_accumulation_steps=gradient_accumulate_every,
            kwargs_handlers=[ddp_kwargs]
        )
        project_name = getattr(cfg, "project_name", 'ResidualDiffsuion-v7')
        self.accelerator.init_trackers(project_name, config=OmegaConf.to_container(cfg, resolve=True))
        create_url_shortcut_of_wandb(accelerator=self.accelerator)
        self.logger = create_logger_of_wandb(accelerator=self.accelerator, rank=not self.accelerator.is_main_process)
        self.accelerator.native_amp = amp
        """
            Initialize the model and parameters.
        """
        self.model = model
        self.train_val_forward_fn = train_val_forward_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.gradient_accumulate_every = gradient_accumulate_every
        # calculate training steps
        self.train_num_epoch = train_num_epoch
        # optimizer
        self.opt = optimizer

        if self.accelerator.is_main_process:
            # save results in wandb folder if results_folder is not specified
            self.results_folder = Path(results_folder if results_folder
                                       else os.path.join(self.accelerator.get_tracker('wandb', unwrap=True).dir, "../"))
            self.results_folder.mkdir(exist_ok=True)
        """
            Initialize the data loader.
        """
        self.cur_epoch = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt, self.scheduler, self.train_loader, self.test_loader \
            = self.accelerator.prepare(self.model, self.opt, scheduler, self.train_loader, self.test_loader)

    def save(self, epoch, max_to_keep=10):
        """
        Delete the old checkpoints to save disk space.
        """
        if not self.accelerator.is_local_main_process:
            return
        ckpt_files = glob.glob(os.path.join(self.results_folder, 'model-[0-9]*.pt'))
        # keep the last n-1 checkpoints
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        ckpt_files_to_delete = ckpt_files[:-max_to_keep]
        for ckpt_file in ckpt_files_to_delete:
            os.remove(ckpt_file)
        data = {
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.model),
            # 'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        save_name = str(self.results_folder / f'model-{epoch}.pt')
        last_save_name = str(self.results_folder / f'model-{epoch}-last.pt')

        # if save file exists, rename it to last_save_name
        if os.path.exists(save_name):
            os.remove(last_save_name) if os.path.exists(last_save_name) else None
            os.rename(save_name, last_save_name)

        torch.save(data, save_name)

    def load(self, resume_path: str = None, pretrained_path: str = None):
        accelerator = self.accelerator
        device = accelerator.device

        if resume_path is not None:
            data = torch.load(resume_path, map_location=device)

            self.cur_epoch = data['epoch']
            # self.opt.load_state_dict(data['opt'])
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

        elif pretrained_path is not None:
            data = torch.load(pretrained_path, map_location=device)
        else:
            raise ValueError('Must specify either milestone or path')
        if self.scheduler is not None:
            # step scheduler to the last epoch
            for _ in range(self.cur_epoch):
                self.scheduler.step()
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'], strict=False)

    @torch.inference_mode()
    @run_on_seed
    def val(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        """
        validation function
        """
        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        model.eval()
        model = accelerator.unwrap_model(model)
        device = model.device
        maes = []
        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
            image, gt, name, image_for_post = data['image'], data['gt'], data['name'], data['image_for_post']
            gt = [np.array(x, np.float32) for x in gt]
            gt = [x / (x.max() + 1e-8) for x in gt]
            image = image.to(device).squeeze(1)
            out = self.train_val_forward_fn(model, image=image, verbose=False)
            res = out["pred"].detach().cpu()
            maes += [cal_mae(g, r, thresholding, save_to, n) for g, r, n in zip(gt, res, name)]
        # gather all the results from different processes
        accelerator.wait_for_everyone()
        mae = accelerator.gather(torch.tensor(maes).mean().to(device))
        mae = mae.mean().item()
        # mae = mae_sum / test_data_loader.dataset.size
        _best_mae = min(_best_mae, mae)
        return mae, _best_mae

    @torch.inference_mode()
    @run_on_seed
    def val_time_ensemble(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        """
        validation function
        """
        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        model.eval()
        model = accelerator.unwrap_model(model)
        device = model.device
        maes = defaultdict(list)
        ensemble_maes = []
        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
            image, gt, name, image_for_post = data['image'], data['gt'], data['name'], data['image_for_post']
            gt = [x.cpu().numpy().astype(np.float32) for x in gt]
            gt = [x / (x.max() + 1e-8) for x in gt]
            image = image.to(device)
            ensem_out = self.train_val_forward_fn(model, image=image, time_ensemble=True,
                                                  gt_sizes=[g.shape[-2:] for g in gt], verbose=False)
            ensem_res = ensem_out["pred"]

            ensemble_maes += [cal_mae(g, r, thresholding, save_to, n) for g, r, n in zip(gt, ensem_res, name)]

        # gather all the results from different processes
        accelerator.wait_for_everyone()
        ensemble_maes = accelerator.gather(torch.tensor(ensemble_maes).mean().to(device)).mean().item()

        _best_mae = min(_best_mae, ensemble_maes)
        return ensemble_maes, _best_mae

    @torch.inference_mode()
    @run_on_seed
    def val_batch_ensemble(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        """
        validation function
        """
        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        model.eval()
        model = accelerator.unwrap_model(model)
        device = model.device
        ensemble_maes = []
        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
            image, gt, name, image_for_post = data['image'], data['gt'], data['name'], data['image_for_post']
            gt = [np.array(x, np.float32) for x in gt]
            gt = [x / (x.max() + 1e-8) for x in gt]
            image = image.to(device)
            batch_res = []
            for i in range(5):
                ensem_out = self.train_val_forward_fn(model, image=image, time_ensemble=True, verbose=False)
                ensem_res = ensem_out["pred"].detach().cpu()
                batch_res.append(ensem_res)
            batch_res = torch.mean(torch.concat(batch_res, dim=1), dim=1, keepdim=True)
            for g, r, n in zip(gt, batch_res, name):
                ensemble_maes.append(cal_mae(g, r, thresholding, save_to, n))

        # gather all the results from different processes
        accelerator.wait_for_everyone()
        ensemble_maes = accelerator.gather(torch.tensor(ensemble_maes).mean().to(device)).mean().item()

        _best_mae = min(_best_mae, ensemble_maes)
        return ensemble_maes, _best_mae

    def train(self):
        accelerator = self.accelerator
        for epoch in range(self.cur_epoch, self.train_num_epoch):
            self.cur_epoch = epoch
            # Train
            self.model.train()
            loss_sm = SmoothedValue(window_size=10)
            with tqdm(total=len(self.train_loader), disable=not accelerator.is_main_process) as pbar:
                for data in self.train_loader:
                    with accelerator.autocast(), accelerator.accumulate(self.model):
                        loss = fill_args_from_dict(self.train_val_forward_fn, data)(model=self.model)
                        accelerator.backward(loss)
                        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.opt.step()
                        self.opt.zero_grad()
                    loss_sm.update(loss.item())
                    pbar.set_description(
                        f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm.avg:.4f}({loss_sm.global_avg:.4f})')
                    self.accelerator.log({'loss': loss_sm.avg, 'lr': self.opt.param_groups[0]['lr']})
                    pbar.update()

                    # if loss_sm.count >= 20:
                    #     break
            if self.scheduler is not None:
                self.scheduler.step()

            accelerator.wait_for_everyone()
            loss_sm_gather = accelerator.gather(torch.tensor([loss_sm.global_avg]).to(accelerator.device))
            loss_sm_avg = loss_sm_gather.mean().item()
            self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm_avg:.4f}')

            # Val
            self.model.eval()
            if (epoch + 1) % 1 == 0 or (epoch >= self.train_num_epoch * 0.7):
                mae, best_mae = self.val_time_ensemble(self.model, self.test_loader, accelerator)
                self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} mae: {mae:.4f}({best_mae:.4f})')
                accelerator.log({'mae': mae, 'best_mae': best_mae})
                if mae == best_mae:
                    self.save("best")
            self.save(self.cur_epoch)

            # Visualize
            with torch.inference_mode():
                if accelerator.is_main_process:
                    model = self.accelerator.unwrap_model(self.model)
                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            out = fill_args_from_dict(self.train_val_forward_fn, data)(model=model,
                                                                                       verbose=False)
                            # Normalize from [-1, 1] to [0, 1] for proper wandb visualization
                            normalized_outputs = {}
                            for key, value in out.items():
                                if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                                    # Clamp to [-1, 1] and convert to [0, 1]
                                    normalized_value = torch.clamp(value, -1, 1)
                                    normalized_value = (normalized_value + 1) / 2
                                    normalized_outputs[key] = normalized_value
                                else:
                                    normalized_outputs[key] = value
                            
                            raw_input_img, prediction_img, ground_truth_img = None, None, None
                            
                            if 'image' in data:
                                input_img = data['image'][0].cpu()
                                input_img = torch.clamp(input_img, -1, 1)
                                input_img = (input_img + 1) / 2
                                raw_input_img = input_img.squeeze().numpy()
                            
                            if 'pred' in normalized_outputs:
                                pred_value = normalized_outputs['pred']
                                prediction_img = composite_onehot_to_mask(pred_value).squeeze().cpu().numpy()
                            
                            if 'gt' in data:
                                gt_tensor = data['gt'][0].cpu()
                                ground_truth_img = composite_onehot_to_mask(gt_tensor).squeeze().cpu().numpy()

                            tracker.log({
                                'validation': [
                                    wandb.Image(raw_input_img, caption="Image"),
                                    wandb.Image(prediction_img, caption="Prediction"), 
                                    wandb.Image(ground_truth_img, caption="Mask")
                                ]
                            })

            accelerator.wait_for_everyone()
        self.logger.info('training complete')
        accelerator.end_training()
