import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from torch.utils.tensorboard import SummaryWriter
from utils.data import convert_image, psnr, cnt_bpp

from model.enc_dec import setup_codec, TruncatedVGG19, freeze_model
from model.sr import UpsampleModel, DownsampleModel, Discriminator

import shutil

class simple_down(nn.Module):
    def __init__(self, sf=2):
        super(simple_down, self).__init__()
        self.i = nn.Identity()
        self.sf = sf
    
    def forward(self, x):
        x = self.i(x)
        return x[..., ::self.sf, ::self.sf]

class BicubicDownsample(nn.Module):
    def __init__(self, sf=2):
        super(BicubicDownsample, self).__init__()
        self.i = nn.Identity()
        self.sf = sf
    
    def forward(self, x):
        x = self.i(x)
        dst_h, dst_w = x.shape[2] // self.sf, x.shape[3] // self.sf
        x = torch.nn.functional.interpolate(x, size=(dst_h, dst_w), mode='bicubic', align_corners=False)
        return x

class TrainingModule(pl.LightningModule):
    def __init__(self, opts, run_path):
        super().__init__()
        self.save_hyperparameters()

        self.use_gan = opts['use_adv']
        if self.use_gan:
            print("\nUSING GAN\n")

        self.use_vgg = opts['use_vgg']
        if self.use_vgg:
            print("\nUSING VGG\n")

        self.use_mse_part = opts['use_mse_part']
        if self.use_mse_part:
            print("\nUSING use_mse_part\n")

        self.use_residual = opts['use_residual']
        if self.use_residual:
            print('\nUSE RESIDUALS')

        if opts['is_simple']:
            self.downsample =       BicubicDownsample(opts['scaling_factor'])
        else:
            self.downsample =       DownsampleModel(**opts['down_params'])


        self.codec =            setup_codec(**opts['codec_params'])
        self.generator =        UpsampleModel(**opts['gen_params'])
        
        if self.use_gan:
            self.discriminator =    Discriminator(**opts['disc_params'])
        
        if self.use_vgg:
            self.vgg =              TruncatedVGG19(**opts['vgg_params'])
            freeze_model(self.vgg)
            self.vgg.eval()

        if self.use_residual:
            self.res_codec = setup_codec(**opts['res_codec_params'])

        self.automatic_optimization = False
        
        # shutil.rmtree(opts['lr_params']['logs_dir'])
        self.writer = SummaryWriter(run_path)

        self.l_mse, self.l_adv, self.l_vgg, self.l_mse_part = (
            opts['lr_params']['l_mse'], 
            opts['lr_params']['l_adv'], 
            opts['lr_params']['l_vgg'],
            opts['lr_params']['l_mse_part'],
        )
        self.n_rows_show = opts['lr_params']['n_rows_show']

    def forward(self, input):
        x = self.downsample(input)
        x = convert_image(x, '[-1, 1]', '[0, 1]')
        x = self.codec(x)['x_hat']
        x = convert_image(x, '[0, 1]', '[-1, 1]')
        x = x.clip(-1, 1)
        x = self.generator(x)
        if self.use_residual:
            pres, _, _ = self.get_residuals(x, input)
            return pres, x 
        
        return x
    
    def get_residuals(self, x_hat, input, get_str = False):
        res = (input - x_hat) / 2 # [-2, 2] to [-1, 1]
        res = convert_image(res, '[-1, 1]', '[0, 1]')
        if get_str:
            t = self.res_codec.compress(res)
            bpp = cnt_bpp(t, torch.prod(torch.tensor(input.shape)))
            res = self.res_codec.decompress(**t)['x_hat']
        else:
            bpp = -1
            res = self.res_codec(res)['x_hat']
        
        res = convert_image(res, '[0, 1]', '[-1, 1]') * 2
        x = x_hat + res 
        x = x.clip(-1, 1)
        return x, res, bpp
    
    def training_step(self, x, batch_idx):
        opt_g = self.optimizers()
        if self.use_gan:
            opt_g, opt_d = opt_g[0], opt_g[1]
        
        # optimize generator
        if self.use_residual:
            x_hat, x_sr = self(x)
        else:
            x_sr = self(x)

        g_loss = 0

        if self.use_residual:
            mse_loss = self.mse_loss(x_hat, x)
            g_loss += self.l_mse * mse_loss
        if self.use_gan:
            g_loss += self.l_adv * self.gan_loss(x_sr, x)
        if self.use_mse_part:
            mse_loss = self.mse_loss(x_sr, x)
            g_loss += self.l_mse_part * mse_loss
        if self.use_vgg:
            vgg_loss = self.vgg_loss(x_sr, x)
            g_loss += self.l_vgg * vgg_loss

        self.log("loss", g_loss, prog_bar=True)
        
        if self.use_residual or self.use_mse_part:
            self.log("mse_loss", mse_loss, prog_bar=True)
        if self.use_gan:
            self.log("g_loss", g_loss, prog_bar=True)
        if self.use_vgg:
            self.log("vgg_loss", vgg_loss, prog_bar=True)

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.writer.add_scalar(f'g_loss', g_loss, batch_idx)

        # optimize discriminator
        if self.use_gan:
            valid = torch.ones(x.size(0), 1).type_as(x)
            fake = torch.zeros(x.size(0), 1).type_as(x)

            real_loss = self.ls_loss(self.discriminator(x), valid)
            
            if self.use_residual:
                x_hat, x_sr = self(x)
            else:
                x_sr = self(x)
            
            fake_loss = self.ls_loss(self.discriminator(x_sr.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)

            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()

            self.writer.add_scalar(f'd_loss', d_loss, batch_idx)

    def gan_loss(self, y_hat, y):
        return nn.functional.binary_cross_entropy_with_logits(y_hat, y)
    
    def vgg_loss(self, y_hat, y):
        y_hat_vgg = self.vgg(y_hat)
        y_vgg = self.vgg(y).detach()
        return nn.functional.mse_loss(y_hat_vgg, y_vgg)

    def ls_loss(self, y_hat, y):
        return 0.5 * ((y_hat - y)**2).mean()

    def mse_loss(self, y_hat, y):
        return nn.functional.mse_loss(y_hat, y)
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            [
                {'params' : p, 'lr' : 2e-4, 'betas' : (0.5, 0.999)}
                for p in [self.downsample.parameters(), self.codec.parameters(), self.generator.parameters()]
            ],
        )
        if self.use_gan:
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=1e-4,
                betas=(0.5, 0.999),
            )
            return [opt_g, opt_d], []
        else:
            return opt_g
    
    def validation_step(self, batch, batch_idx):
        x = batch
        if self.use_residual:
            x_hat, x_sr = self(x)
        else:
            x_hat = self(x)

        loss = psnr(x_hat, x)

        self.log("psnr", loss, prog_bar=True)
        self.writer.add_scalar(f'psnr', loss, batch_idx)

        self.last_valid_img = x

        return loss

    def on_validation_epoch_end(self):
        idx = torch.randperm(self.last_valid_img.shape[0])[:4]
        x = self.last_valid_img[idx]

        with torch.no_grad():
            x_down = self.downsample(x)
            x_down = convert_image(x_down, '[-1, 1]', '[0, 1]')
            x_dec = self.codec(x_down)['x_hat']
            x_dec = convert_image(x_dec, '[0, 1]', '[-1, 1]')
            x_dec = x_dec.clip(-1, 1)
            x_ups = self.generator(x_dec)

            x_down = convert_image(x_down, '[0, 1]', '[-1, 1]')
            
            if self.use_residual:
                x_rec, res, _ = self.get_residuals(x_ups, x)

                x_down = torchvision.transforms.Resize(size=x.size()[2:])(x_down)
                x_dec = torchvision.transforms.Resize(size=x.size()[2:])(x_dec)

                img = torch.zeros(x.shape[0] * 6, *x.shape[1:])
                img[:: 6] = x
                img[1::6] = x_down
                img[2::6] = x_dec
                img[3::6] = x_ups
                img[4::6] = res
                img[5::6] = x_rec
                val_grid = torchvision.utils.make_grid(img.cpu(), nrow=6)
            else:
                x_down = torchvision.transforms.Resize(size=x.size()[2:])(x_down)
                x_dec = torchvision.transforms.Resize(size=x.size()[2:])(x_dec)

                img = torch.zeros(x.shape[0] * 4, *x.shape[1:])
                img[:: 4] = x
                img[1::4] = x_down
                img[2::4] = x_dec
                img[3::4] = x_ups
                val_grid = torchvision.utils.make_grid(img.cpu(), nrow=4)
            
            self.writer.add_image('GT/LQ/DEC/SR/RES/REC', val_grid, global_step=self.current_epoch)
        
    def test_step(self, batch, batch_idx):
        x = batch

        if hasattr(self, 'test_dict') == False:
            self.test_dict = {'psnr' : [], 'bpp1' : [], 'bpp2' : [], 'bpp' : []}

        n_pix = torch.prod(torch.tensor(x.shape))

        x_down = self.downsample(x)
        x_down = convert_image(x_down, '[-1, 1]', '[0, 1]')
        cod1 = self.codec.compress(x_down)
        bpp1 = cnt_bpp(cod1, n_pix)
        dec1 = self.codec.decompress(**cod1)['x_hat']
        dec1 = convert_image(dec1, '[0, 1]', '[-1, 1]')
        ups = self.generator(dec1)
        
        bpp2 = torch.tensor([0])
        if self.use_residual:
            ups = convert_image(ups, '[-1, 1]', '[0, 1]')
            ups, _, bpp2 = self.get_residuals(ups, x, True)

        test_bpp = bpp1 + bpp2
        print(torch.isnan(ups).sum())
        print(torch.isnan(x).sum())
        test_loss = psnr(ups, x)
        self.test_dict['psnr'].append(test_loss.item())
        self.test_dict['bpp1'].append(bpp1.item())
        self.test_dict['bpp2'].append(bpp2.item())
        self.test_dict['bpp'].append(test_bpp.item())

        self.log("test_psnr", test_loss)
        self.log("test_bpp",  test_bpp)
        
        return test_loss