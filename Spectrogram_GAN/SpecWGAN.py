#Reference: https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
#Reference: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/generative_adversarial_network/main.py
import os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import grad
from torch.autograd import Variable
import tensorboardX
import phasesuffle as P
import checkpoint

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# the folder of results
name = 'NSynth_1'
out_folder = './Spec_WGAN_results/'
if not os.path.isdir(out_folder):
    os.mkdir(out_folder)

out_folder = out_folder + name
if not os.path.isdir(out_folder):
    os.mkdir(out_folder)
if not os.path.isdir(out_folder + '/Random_results'):
    os.mkdir(out_folder + '/Random_results')
if not os.path.isdir(out_folder + '/Fixed_results'):
    os.mkdir(out_folder + '/Fixed_results')



d=16
c=1

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.linear = nn.Linear(100, d*256)

        self.deconv1 = nn.ConvTranspose2d(d*16, d*8, (4,4), (2,2), (1,1))
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, (4,4), (2,2), (1,1))
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, (4,4), (2,2), (1,1))
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, (4,4), (2,2), (1,1))
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, (4,4), (2,2), (1,1))

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = self.linear(input)
        x = x.reshape(x.size(0), d*16, 4, 4)
        x = F.relu(x)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, (4,4), (2,2), (1,1))
        self.conv2 = nn.Conv2d(d, d*2, (4,4), (2,2), (1,1))
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, (4,4), (2,2), (1,1))
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, (4,4), (2,2), (1,1))
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, d*16, (4,4), (2,2), (1,1))

        self.linear = nn.Linear(d*256, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = x.reshape(x.size(0), d*256)
        x = self.linear(x)

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()



##############################
########   UTILITIES  ########
##############################

#fixed_z = torch.randn(10, 100).view(-1, 100).to(device)

#fixed_z_ = torch.randn(5*5, 100).view(-1, 100).to(device)    # fixed noise
#fixed_z_ = torch.FloatTensor(5*5, 100).uniform_(-1, 1).view(-1, 100).to(device)
#np.save(out_folder + '/fixed_z.npy', fixed_z.cpu().numpy())

fixed_z = torch.from_numpy(np.load(out_folder + '/fixed_z.npy')).to(device)

def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_gp']))

    y1 = hist['D_gp']
    y2 = hist['D_wd']
    y3 = hist['G_losses']

    # Discriminator
    plt.close('all')

    plt.plot(x, y1, label='D_gp')
    plt.plot(x, y2, label='D_wd')
    plt.plot(x, y3, label='D_losses')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path+'_dis.png')

    if show:
        plt.show()
    else:
        plt.close()

    # Generator
    plt.close('all')

    plt.plot(x, y3, label='G_losses')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path + '_gen.png')

    if show:
        plt.show()
    else:
        plt.close()



################################
##########   SETTING   #########
################################

batch_size = 64
lr = 0.0001
train_epoch = 180

# data_loader
dataset = torchvision.datasets.DatasetFolder(root='./spec_norm', loader = np.load, extensions = ['.npy'])
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, drop_last= True)

# network
G = generator()
D = discriminator()
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G = G.to(device)
D = D.to(device)

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

(...)

def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)

    # gradient penalty
    z = Variable(z, requires_grad=True).cuda()
    o = f(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda()
                 , create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

    return gp


###################################
#######    TRAINING PART   ########
###################################

writer = tensorboardX.SummaryWriter('./summaries/AudioWGAN')

print('training start!')
start_time = time.time()
for epoch in range(start_epoch, train_epoch):
    D_gp = []
    D_wd = []
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for i, (x_, _) in enumerate(dataloader):
        x_ = x_.type(torch.cuda.FloatTensor)
        step = epoch * len(dataloader) + i + 1

        if (1):
            # train discriminator D
            D.zero_grad()
            x_ = x_.to(device)
            x_ = x_.unsqueeze(1)
            z_ = torch.randn((batch_size, 100)).view(-1, 100).to(device)
            fx_ = G(z_)

            r_logit = D(x_)
            f_logit = D(fx_)

            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = gradient_penalty(x_.data, fx_.data, D)
            D_train_loss = -wd + gp * 10.0

            D_train_loss.backward()
            D_optimizer.step()

            D_gp.append(gp.data)
            D_wd.append(wd.data)
            D_losses.append(D_train_loss.data)

            writer.add_scalar('D/wd', wd.data.cpu().numpy(), global_step=step)
            writer.add_scalar('D/gp', gp.data.cpu().numpy(), global_step=step)
            writer.add_scalar('D_loss', D_train_loss.data.cpu().numpy()
                                  , global_step=step)

        if (step % 5 == 0):
            # train generator G
            G.zero_grad()

            z_ = torch.randn((batch_size, 100)).view(-1, 100).to(device)

            fx_ = G(z_)
            f_logit = D(fx_)
            G_train_loss = -f_logit.mean()

            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data)

            writer.add_scalars('G',
                               {"G_loss": G_train_loss.data.cpu().numpy()}
                               , global_step=step)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
    (epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
    torch.mean(torch.FloatTensor(G_losses))))

    if ((epoch + 1) % 10 == 0):
        p = out_folder + '/Random_results/epoch_' + str(epoch + 1)
        fixed_p = out_folder + '/Fixed_results/epoch_' + str(epoch + 1)

        if not os.path.isdir(p):
            os.mkdir(p)
        if not os.path.isdir(fixed_p):
            os.mkdir(fixed_p)

        show_result((epoch + 1), save=True, path=p, isFix=False)
        show_result((epoch + 1), save=True, path=fixed_p, isFix=True)

    train_hist['D_gp'].append(torch.mean(torch.FloatTensor(D_gp)))
    train_hist['D_wd'].append(torch.mean(torch.FloatTensor(D_wd)))
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    show_train_hist(train_hist, show=True, save=True, path=out_folder + '/Spec_WGAN_train_hist.png')

    checkpoint.save_checkpoint({'epoch': epoch + 1,
                           'D': D.state_dict(),
                           'G': G.state_dict(),
                           'd_optimizer': D_optimizer.state_dict(),
                           'g_optimizer': G_optimizer.state_dict()},
                          '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                          max_keep=5)


end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), out_folder + '/generator_param.pkl')
torch.save(D.state_dict(), out_folder + '/discriminator_param.pkl')
torch.save(G_optimizer.state_dict(), out_folder + '/g_optimizer_param.pkl')
torch.save(D_optimizer.state_dict(), out_folder + '/d_optimizer_param.pkl')

with open(out_folder + '/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=out_folder + '/Spec_WGAN_train_hist.png')

'''
images = []
for epoch in range(train_epoch):
    if((epoch+1)%100 == 0):
        img_name = out_folder + '/Fixed_results/epoch_' + str(epoch + 1) + '.png'
        images.append(imageio.imread(img_name))
imageio.mimsave(out_folder + '/generation_animation.gif', images, fps=5)