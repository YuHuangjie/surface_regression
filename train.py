import os, cv2
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

def run_epoch(model, train_dataloader, writer, optim, pbar, epoch, total_steps):

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

    return total_steps

def run_val(model, val_dataloader, pbar, writer, epoch):

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

            writer.add_scalar("val_loss", torch.mean(torch.Tensor(val_losses)), epoch)
            pbar.set_description((f"val_loss {torch.mean(torch.Tensor(val_losses)):.4f}"))

def run_test(model, test_dataloader, writer, logdir, test_set, epoch):

    # make full testing
    tqdm.write("Running full validation set...")
    output_dir = os.path.join(logdir, 'result')
    os.makedirs(output_dir, exist_ok=True)
    psnr = []
    dsize = (test_set.H, test_set.W)

    with torch.no_grad():
        for i, (x, residual, mask, approx) in enumerate(test_dataloader):
            x, residual = x.cuda(), residual.cuda()
            y = model_pred(model, x)
            img = torch.zeros((dsize[0] * dsize[1], 3))
            img[mask[0]] = y.cpu() + approx[0].cpu()
            img = np.clip(img.numpy() * 255., 0, 255).astype(np.uint8)
            img = img.reshape(dsize + (3,))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            assert (cv2.imwrite(f'{output_dir}/{i}.png', img))
            psnr.append(model_psnr(model_loss(y, residual)).item())

    writer.add_scalar('test_psnr', np.mean(psnr), epoch)
    writer.add_image('test_img', img, epoch, dataformats='HWC')

    # save test psnrs
    np.savetxt(f'{output_dir}/test_{epoch}_psnr.txt', psnr, newline=',\n')

