
import numpy as np
import torch
from torch.autograd import Variable
def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sum=True, sigma=None):
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c)))
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))

    Z = torch.sqrt(torch.tensor(2.0 / num_f))

    mid = torch.matmul(x, w.t())

    mid = mid + b
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0]
    mid *= np.pi / 2.0
    if sum:
        Z = Z * (torch.cos(mid)+ torch.sin(mid))# 
    else:
        Z = Z * torch.cat((torch.cos(mid), torch.sin(mid)), dim=-1)#
    return Z      # 得到返回的张量z

def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        cov = torch.matmul((w * x).t(), x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    return res

def lossb_expect0(cfeaturec, weight, num_f, sum=True):
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, sum=sum)
    loss = Variable(torch.FloatTensor([0]))
    weight = weight
    for i in range(cfeaturecs.size()[-1]):
        cfeaturec = cfeaturecs[:, :, i]
        cov1 = torch.cov(cfeaturec.t())
        cov_matrix = cov1 * cov1
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
    return loss

def lossb_expect(cfeaturec, weight, num_f, sum=True):
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, sum=sum)
    loss = Variable(torch.FloatTensor([0]))
    weight = weight
    for i in range(cfeaturecs.size()[-1]):
        cfeaturec = cfeaturecs[:, :, i]
        cov1 = cov(cfeaturec, weight)
        cov_matrix = cov1 * cov1
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
    return loss
def lossb_expect1(cfeaturec):
    cfeaturecs=cfeaturec
    loss = Variable(torch.FloatTensor([0]))
    cfeaturecs0=cfeaturecs.transpose(1,2)
    cov1= torch.matmul(cfeaturecs,cfeaturecs0)
    cov_matrix0=cov1 * cov1  
    for i in range(cfeaturecs.size()[1]):
        cov_matrix = cov_matrix0[i]
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
    return loss