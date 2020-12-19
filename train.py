import os
import shutil
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

model_pred = lambda model, x: model(x)
model_loss = lambda pred, y: torch.mean((pred - y) ** 2)
model_loss2 = lambda model, x, y: torch.mean((model_pred(model, x) - y) ** 2)
model_psnr = lambda loss : -10. * torch.log10(loss)

def train(model, train_dataloader, lr, epochs, logdir, epochs_til_checkpoint=10, 
    steps_til_summary=100, val_dataloader=None, global_step=0, model_params=None):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    os.makedirs(logdir, exist_ok=True)

    checkpoints_dir = os.path.join(logdir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    summaries_dir = os.path.join(logdir,  'summaries')
    os.makedirs(summaries_dir, exist_ok=True)

    writer = SummaryWriter(summaries_dir, purge_step=global_step)

    total_steps = global_step
    pbar = tqdm(range(epochs), dynamic_ncols=True, smoothing=0.01)
    for epoch in pbar:

        model.train()
        for step, (model_input, gt, *_) in enumerate(train_dataloader):

            model_input = model_input.cuda()
            gt = gt.cuda()

            train_loss = model_loss2(model, model_input, gt)
            writer.add_scalar('train_loss', train_loss.item(), total_steps)
            writer.add_scalar('train_psnr', model_psnr(train_loss).item(), total_steps)

            optim.zero_grad()
            train_loss.backward()
            optim.step()

            # evaludate
            pbar.set_description((f"Epoch {epoch}, Total loss {train_loss.item():.6}, PSNR {model_psnr(torch.Tensor([train_loss.item()])).item():.2f}"))

            total_steps += 1

        if not epoch % epochs_til_checkpoint and epoch:
            torch.save({'model': model.state_dict(),
                        'params': model_params,
                        'global_step': total_steps},
                       os.path.join(checkpoints_dir, f'model_epoch_{epoch:04}.pt'))

            if val_dataloader is not None:
                pbar.set_description(("Running partial validation set..."))
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    for (model_input, gt, *_) in val_dataloader:
                        model_input, gt = model_input.cuda(), gt.cuda()
                        val_loss = model_loss2(model, model_input, gt)
                        val_losses.append(val_loss)
                        if len(val_losses) > 10:
                            break

                    writer.add_scalar("val_loss", torch.mean(torch.Tensor(val_losses)), total_steps)
                    pbar.set_description((f"val_loss {torch.mean(torch.Tensor(val_losses)):.4f}"))

    torch.save({'model': model.state_dict(),
                'params': model_params,
                'global_step': total_steps},
               os.path.join(checkpoints_dir, 'model_final.pt'))
