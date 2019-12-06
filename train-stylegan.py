import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
from models.stylegan import StyleGanGenerator, StyleGanDiscriminator
import torchvision_sunner.transforms as trans
import torchvision_sunner.data as dataset
from losses import gradient_penalty, r1_penalty, r2_penalty
from opts import train_opts
from torchvision import utils
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import numpy as np
import math
import os

logger = logging.getLogger('Style GAN')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
handler_format = logging.Formatter(
    '[%(module)-10s (%(levelname)-8s) - %(asctime)s] %(message)s'
)
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)
torch.manual_seed(42)


def train(opts):
    def log(string, name="stylegan.log"):
        with open(name, 'a') as f:
            f.write(string + '\n')
    writer = SummaryWriter(str(opts.output))
    loader = dataset.DataLoader(
        dataset=dataset.ImageDataset(
            [[opts.input]],
            transform=transforms.Compose([
                trans.Resize((opts.imsize, opts.imsize)),
                trans.ToTensor(),
                trans.ToFloat(),
                trans.Transpose(trans.BHWC2BCHW),
                trans.Normalize()
            ])
        ),
        batch_size=opts.batch_size,
        shuffle=True
    )
    G = opts.G
    D = opts.D
    step = 0
    start_epoch = opts.start_epoch
    if opts.resume:
        try:
            assert os.path.exists(opts.resume)
            state = torch.load(opts.resume)
            G.load_state_dict(state['G'])
            D.load_state_dict(state['D'])
            start_epoch = state['start_epoch']
            logger.info("Load Pretrained Weight")
        except:
            logger.warn("Resume Files cannot Load")
            logger.info("Train from Scratch")
    else:
        logger.info("Train from Scratch")
    if torch.cuda.device_count() > 1 and opts.device == 'cuda':
        logger.info(f"{torch.cuda.device_count()} GPUs found.")
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
    if opts.device == 'cuda':
        torch.backends.cudnn.benchmark = True
    G.to(opts.device)
    D.to(opts.device)
    optimG = Adam(G.parameters(), lr=opts.g_lr, betas=opts.betas)
    optimD = Adam(D.parameters(), lr=opts.d_lr, betas=opts.betas)
    schedulerG = lr_scheduler.ExponentialLR(optimG, gamma=opts.g_lrdecay)
    schedulerD = lr_scheduler.ExponentialLR(optimD, gamma=opts.d_lrdecay)
    fixed_z = torch.randn([opts.batch_size, 512]).to(opts.device)
    sp = nn.Softplus()
    g_store = [0.0]
    d_store = [0.0]
    for epoch in range(start_epoch, opts.epochs + 1):
        bar = tqdm(loader)
        glosses = []
        dlosses = []
        for i, (real, ) in enumerate(bar):
            step += 1
            D.zero_grad()
            real = real.to(opts.device)
            Dr = D(real)
            writer.add_graph(D, real)
            z = torch.randn([real.size(0), 512]).to(opts.device)
            fake = G(z)
            writer.add_graph(G, z)
            Df = D(fake.detach())
            Dloss = sp(Df).mean() + sp(-Dr).mean()
            if opts.r1gamma > 0:
                r1 = r1_penalty(real.detach(), D)
                Dloss = Dloss + r1 * (opts.r1gamma * .5)
            if opts.r2gamma > 0:
                r2 = r2_penalty(fake.detach(), D)
                Dloss = Dloss + r2 * (opts.r2gamma * .5)
            dlosses.append(Dloss.item())
            Dloss.backward()
            optimD.step()
            if i % opts.critic_iters == 0:
                G.zero_grad()
                Df = D(fake)
                Gloss = sp(-Df).mean()
                glosses.append(Gloss.item())
                Gloss.backward()
                optimG.step()
            if i % opts.show_interval == 0:
                with torch.no_grad():
                    nr = int(math.ceil(math.sqrt(opts.batch_size)))
                    z = torch.randn([real.size(0), 512]).to(opts.device)
                    img = G(z)
                    save_image(
                        img.detach().cpu(),
                        os.path.join(opts.output, 'images',
                                     'normal', f'{epoch:04}_{i:06}.png'),
                        nrow=nr,
                        normalize=True
                    )
                    fakes = utils.make_grid(img, nr, padding=0)
                    fakes = fakes.to(torch.float32).cpu().numpy()
                    fakes = np.clip((fakes / 2) + 0.5, 0, 1)
                    writer.add_image(
                        f"EPOCH{epoch}/Random", torch.from_numpy(fakes), i)
                    img = G(fixed_z)
                    save_image(
                        img.detach().cpu(),
                        os.path.join(opts.output, 'images',
                                     'fixed', f'{epoch:04}_{i:06}.png'),
                        nrow=nr,
                        normalize=True
                    )
                    fakes = utils.make_grid(img, nr, padding=0)
                    fakes = fakes.to(torch.float32).cpu().numpy()
                    fakes = np.clip((fakes / 2) + 0.5, 0, 1)
                    writer.add_image(
                        f"EPOCH{epoch}/Fixed", torch.from_numpy(fakes), i)
            writer.add_scalar(f"LOSS/Generator",
                              Gloss.item(), global_step=step)
            writer.add_scalar(f"LOSS/Discriminator",
                              Dloss.item(), global_step=step)
            bar.set_description(
                f"Epoch {epoch}/{opts.epochs} G: {glosses[-1]:.6f} D: {dlosses[-1]:.6f}"
            )
        g_store.append(np.mean(glosses))
        d_store.append(np.mean(dlosses))
        state = {
            'G': G.state_dict(),
            'D': D.state_dict(),
            'Loss_G': g_store,
            'Loss_D': d_store,
            'start_epoch': epoch,
            'opts': opts
        }
        torch.save(state, os.path.join(opts.output, 'models', 'latest.pth'))
        if epoch % 10 == 0:
            torch.save(
                state, os.path.join(opts.output, 'models', f'{epoch:04}.pth')
            )
        schedulerD.step()
        schedulerG.step()


if __name__ == '__main__':
    opts = train_opts.TrainParser()
    train(opts)
