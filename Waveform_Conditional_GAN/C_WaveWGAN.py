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
name = 'drums_1'
out_folder = './Wave_C_WGAN_results/'
if not os.path.isdir(out_folder):
    os.mkdir(out_folder)

out_folder = out_folder + name
if not os.path.isdir(out_folder):
    os.mkdir(out_folder)
if not os.path.isdir(out_folder + '/Random_results'):
    os.mkdir(out_folder + '/Random_results')
if not os.path.isdir(out_folder + '/Fixed_results'):
    os.mkdir(out_folder + '/Fixed_results')


##############################
#######  NETWORK PART  #######
##############################

d=64 # model size
c=1 # audio channel num
rad = 2 # phase shuffle size

dim_latent = 100
num_label = 8

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.linear_1 = nn.Linear(dim_latent, d*128)
        self.linear_2 = nn.Linear(num_label, d*128)
        self.convtr1 = nn.ConvTranspose1d(d*16, d*8, 25, 4, 11, 1)
        self.convtr1_bn = nn.BatchNorm1d(d*8)
        self.convtr2 = nn.ConvTranspose1d(d*8, d*4, 25, 4, 11, 1)
        self.convtr2_bn = nn.BatchNorm1d(d*4)
        self.convtr3 = nn.ConvTranspose1d(d*4, d*2, 25, 4, 11, 1)
        self.convtr3_bn = nn.BatchNorm1d(d*2)
        self.convtr4 = nn.ConvTranspose1d(d*2, d, 25, 4, 11, 1)
        self.convtr4_bn = nn.BatchNorm1d(d)
        self.convtr5 = nn.ConvTranspose1d(d, c, 25, 4, 11, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = self.linear_1(input)
        x = x.reshape(x.size(0), d*8, 16)
        y = self.linear_2(label)
        y = y.reshape(y.size(0), d*8, 16)
        x = torch.cat([x, y], 1)

        x = F.relu(x)
        x = F.relu(self.convtr1(x))
        x = F.relu(self.convtr2(x))
        x = F.relu(self.convtr3(x))
        x = F.relu(self.convtr4(x))
        x = torch.tanh(self.convtr5(x))
        '''
        x = F.relu(self.convtr1_bn(self.convtr1(x)))
        x = F.relu(self.convtr2_bn(self.convtr2(x)))
        x = F.relu(self.convtr3_bn(self.convtr3(x)))
        x = F.relu(self.convtr4_bn(self.convtr4(x)))
        x = torch.tanh(self.convtr5(x))
        '''
        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1_1 = nn.Conv1d(c, int(d/2), 25, 4, 11)
        self.conv1_2 = nn.Conv1d(num_label, int(d/2), 25, 4, 11)
        self.conv2 = nn.Conv1d(d, d*2, 25, 4, 11)
        self.conv2_bn = nn.BatchNorm1d(d*2)
        self.conv3 = nn.Conv1d(d*2, d*4, 25, 4, 11)
        self.conv3_bn = nn.BatchNorm1d(d*4)
        self.conv4 = nn.Conv1d(d*4, d*8, 25, 4, 11)
        self.conv4_bn = nn.BatchNorm1d(d*8)
        self.conv5 = nn.Conv1d(d*8, d*16, 25, 4, 11)
        self.linear = nn.Linear(d*256, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)

        x = P.PhaseShuffle(x, rad)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = P.PhaseShuffle(x, rad)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = P.PhaseShuffle(x, rad)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = P.PhaseShuffle(x, rad)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = x.reshape(x.size(0), d * 256)
        x = self.linear(x)

        '''
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = x.reshape(x.size(0), d*256)
        x = self.linear(x)
        '''

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


##############################
########   UTILITIES  ########
##############################

# label preprocess
onehot_g = torch.zeros(num_label, num_label)
for i in range(num_label):
    onehot_g[i, i] = 1
#onehot_g = onehot_g.scatter_(1, torch.LongTensor(range(num_label).view(num_label, 1)), 1)

onehot_d = torch.zeros(num_label, num_label, 16384)
for i in range(num_label):
    onehot_d[i, i, :] = 1


#fixed_z = torch.randn(10, 100).view(-1, 100).to(device)

#fixed_z_ = torch.randn(5*5, 100).view(-1, 100).to(device)    # fixed noise
#fixed_z_ = torch.FloatTensor(5*5, 100).uniform_(-1, 1).view(-1, 100).to(device)
#np.save(out_folder + '/fixed_z.npy', fixed_z.cpu().numpy())

fixed_z = torch.from_numpy(np.load(out_folder + '/fixed_z.npy')).to(device)

def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z = torch.randn((10, 100)).view(-1, 100).to(device)

    G.eval()

    label = torch.zeros(10, 1)

    for j in range(num_label):

        if not os.path.isdir(path + '/label_{0}'.format(j)):
            os.mkdir(path + '/label_{0}'.format(j))

        label = label.type(torch.LongTensor).squeeze()
        label_g = onehot_g[label].to(device)

        if isFix:
            test_images = G(fixed_z, label_g)
            for i in range(test_images.size(0)):
                np.save(path+'/label_{0}/{1}.npy'.format(j, i), test_images.detach().cpu().numpy()[i])
        else:
            test_images = G(z, label_g)
            for i in range(test_images.size(0)):
                np.save(path+'/label_{0}/{1}.npy'.format(j, i), test_images.detach().cpu().numpy()[i])

        label = label + 1

    G.train()

    '''
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
    '''

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


##############################
##########  SETTING ##########
##############################
batch_size = 64
lr = 0.0001
train_epoch = 25300

# data_loader
dataset = torchvision.datasets.DatasetFolder(root='./feature_c_drums', loader = np.load, extensions = ['.npy'])
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

""" load checkpoint """
ckpt_dir = './Wave_C_WGAN_chekpoints/'
if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)

ckpt_dir = ckpt_dir + name
if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)

try:
    ckpt = checkpoint.load_checkpoint(ckpt_dir)
    start_epoch = ckpt['epoch']
    D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    D_optimizer.load_state_dict(ckpt['d_optimizer'])
    G_optimizer.load_state_dict(ckpt['g_optimizer'])
except:
    print(' [*] No checkpoint!')
    start_epoch = 0

train_hist = {}
train_hist['D_gp'] = []
train_hist['D_wd'] = []
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []


###################
#######  GP #######
###################

def gradient_penalty(x, y, label, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).to(device)
    z = x + alpha * (y - x)

    # gradient penalty
    z = Variable(z, requires_grad=True).to(device)
    o = f(z, label)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).to(device), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

    return gp


##############################
#######  TRAINING PART #######
##############################

writer = tensorboardX.SummaryWriter(out_folder + '/summaries/')

print('training start!')
start_time = time.time()
for epoch in range(start_epoch, train_epoch):
    D_gp = []
    D_wd = []
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for i, (x, label_x) in enumerate(dataloader):
        x = x.type(torch.cuda.FloatTensor)
        step = epoch * len(dataloader) + i + 1
        #print(step)
        #print(x.shape)
        if (1):
            # train discriminator D
            D.zero_grad()
            x = x.to(device)
            x = x.unsqueeze(1)
            label_x_d = onehot_d[label_x].to(device)
            label_x_g = onehot_g[label_x].to(device)

            z = torch.randn((batch_size, 100)).view(-1, 100).to(device)
            #z = torch.FloatTensor(batch_size, 100).uniform_(-1, 1).view(-1, 100).to(device)

            #label_z = torch.FloatTensor(batch_size, 1).uniform_(0, num_label).type(torch.LongTensor).squeeze()
            #label_z_g = onehot_g[label_z]
            #label_z_d = onehot_d[label_z]
            label_z_d = label_x_d.to(device)
            label_z_g = label_x_g.to(device)

            Gz = G(z, label_z_g)

            r_logit = D(x, label_x_d)
            f_logit = D(Gz, label_z_d)

            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = gradient_penalty(x.data, Gz.data, label_x_d.data, D)
            D_train_loss = -wd + gp * 10.0

            D_train_loss.backward()
            D_optimizer.step()

            D_gp.append(gp.data)
            D_wd.append(wd.data)
            D_losses.append(D_train_loss.data)

            writer.add_scalar('D/wd', wd.data.cpu().numpy(), global_step=step)
            writer.add_scalar('D/gp', gp.data.cpu().numpy(), global_step=step)
            writer.add_scalar('D_loss', D_train_loss.data.cpu().numpy(), global_step=step)

        if (step % 5 == 0):
            # train generator G
            G.zero_grad()

            z = torch.randn((batch_size, 100)).view(-1, 100).to(device)
            #z_ = torch.FloatTensor(batch_size, 100).uniform_(-1, 1).view(-1, 100).to(device)
            label_z = torch.FloatTensor(batch_size, 1).uniform_(0, num_label).type(torch.LongTensor).squeeze()
            #label_z = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).squeeze()
            label_z_g = onehot_g[label_z].to(device)
            label_z_d = onehot_d[label_z].to(device)

            Gz = G(z, label_z_g)
            f_logit = D(Gz, label_z_d)
            G_train_loss = -f_logit.mean()

            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data)

            writer.add_scalars('G',
                               {"G_loss": G_train_loss.data.cpu().numpy()},
                               global_step=step)

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
    show_train_hist(train_hist, show=True, save=True, path=out_folder + '/Audio_WGAN_train_hist.png')

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

show_train_hist(train_hist, save=True, path=out_folder + '/Audio_C_WGAN_train_hist.png')

'''
images = []
for epoch in range(train_epoch):
    if((epoch+1)%100 == 0):
        img_name = out_folder + '/Fixed_results/epoch_' + str(epoch + 1) + '.png'
        images.append(imageio.imread(img_name))
imageio.mimsave(out_folder + '/generation_animation.gif', images, fps=5)
'''
