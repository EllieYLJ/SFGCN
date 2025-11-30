import _pickle as pickle
import numpy as np
import torch.utils.data as Data
import argparse
import math
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from utils import ldata
from torch.autograd import Variable
import torch
from models import DenseGCNConv1
import torch.nn.functional as F
from loss import lossb_expect0
import time

###设定参数
parser = argparse.ArgumentParser(description='train DEAP')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_true', help='use CUDA (default: True)')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit (default: 100)')
parser.add_argument('--lr', type=float, default= 1e-3,help='initial learning rate (default: 2e-4)')
parser.add_argument('--seed', type=int, default=9999,help='random seed (default: 9999)')
args = parser.parse_args()
torch.manual_seed(args.seed)
device=torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")#
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
#定义网络架构
class gcn1(torch.nn.Module):
    def __init__(self,n):
        super(gcn1,self).__init__()
        self.w = torch.nn.Parameter(torch.zeros(1,3))
        self.gconv1 = DenseGCNConv1(128,32)
        self.gconv2 = DenseGCNConv1(128,32)
        self.gconv3 = DenseGCNConv1(128,32)
        self.mlp = torch.nn.Linear(32*8*3,32)######！！！！！！！deap=32，dreamer=14
        self.mlp1 = torch.nn.Linear(32,2)

        self.fc1 = torch.nn.Linear(3,16)
        self.fc2 = torch.nn.Linear(16,3)
        torch.nn.init.xavier_uniform_(self.w)
    def forward(self,x,A,tra):
        x1=F.relu(self.gconv1 (x,A[:,:,0]))
        x2=F.relu(self.gconv2 (x,A[:,:,1]))
        x3=F.relu(self.gconv3 (x,A[:,:,2]))

        x1=torch.unsqueeze(x1,dim=1)
        x2=torch.unsqueeze(x2,dim=1)
        x3=torch.unsqueeze(x3,dim=1)
        xa=torch.cat((x1,x2),dim=1)
        xa=torch.cat((xa,x3),dim=1)
        xa=xa.view(xa.size(0),xa.size(1),-1)

        xb=xa.mean(2)
        w1=F.relu(self.fc1(xb))

        w2=torch.sigmoid(self.fc2(w1))
        w2=torch.unsqueeze(w2,dim=2)
        xb=torch.mul(xa,w2)
        x14 = xb.view(xb.size(0), -1)
        x40 = F.relu(self.mlp(x14))
        out =  self.mlp1(x40)
        pred = F.softmax(out,1)
        return out,pred,x40,w2

def train(model, epoch,train_load,a0,lossf, optimizer):
    model.train()
    loss_all = 0
    for train_idx,(data, target) in enumerate(train_load):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output,p,x40,_ = model(data,a0,1)
        loss = lossf(p,target)
        loss.backward()
        loss_all += loss
        optimizer.step()
    return loss_all/len(train_load),loss
def evaluate(model,epoch,test_load,a0,lossf):
    model.eval()
    with torch.no_grad():
       test_loss=0
       predictions0 = []
       labels=[]
       for data, target in test_load:
            data, target = Variable(data), Variable(target)
            label = target.detach().cpu().numpy()
            output, pred,_,w = model(data,a0,0)
            loss = lossf(output, target)
            test_loss += loss
            pred = pred.detach().cpu().numpy()
            pred = np.squeeze(pred)
            predictions0.append(pred)
            labels.append(label)
       predictions0 = np.vstack(predictions0)
       predictions = np.argmax(predictions0, axis = -1)
       labels = np.vstack(labels)####
       labels = np.argmax(labels, axis = -1)####
       acc = accuracy_score(labels, predictions)
    return w,acc
def trainb(model, epoch,train_load,a0,lossf, optimizer):
    model.train()
    loss_all = 0
    for train_idx,(data, target) in enumerate(train_load):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output,p,x40,_ = model(data,a0,1)###
        lossb = lossb_expect0(x40,None,1, True)
        loss = lossf(p,target)+lossb
        loss.backward()
        loss_all += loss
        optimizer.step()
    return loss_all/len(train_load),lossb

def PDC(x, f1):
    x1 = (x[:, :-1]).T
    y = (x[:, 1:]).T
    a = np.dot(np.dot(np.linalg.inv(np.dot(x1.T, x1)), x1.T), y)
    i = np.identity(x.shape[0])
    mb = np.zeros((x.shape[0], x.shape[0]))
    mc = np.zeros((x.shape[0], x.shape[0]))
    for f in range(1, f1):
        af = i - a * (np.exp(-2j * math.pi * f))
        aff = np.dot(np.linalg.inv(af), af)
        af1 = abs(af) / abs(aff)
        row, col = np.diag_indices_from(af1)
        af1[row, col] = 0
        b = af1 / np.max(af1)
        b[row, col] = 1
        mb += b

        af2 = abs(np.linalg.inv(af)) ** 2
        af2[row, col] = 0
        c = af2 / np.max(af1)
        c[row, col] = 1
        mc += c
    return mb / (f1 - 1), mc / (f1 - 1)
def nPDC (x):
    n=x.shape[0]
    c=x.shape[1]
    b1=np.zeros((n,c,c))
    c1=np.zeros((n,c,c))
    for i in range (n):
        x1=x[i]
        b,c=PDC(x1,40)
        b1[i]=b
        c1[i]=c
    mb=np.mean(b1,axis=0)
    mc=np.mean(c1,axis=0)
    p1=np.percentile(mb,80)
    mb[mb<p1]=0
    p2=np.percentile(mc,80)
    mc[mc<p2]=0
    return mb,mc
def PCC(x):
  l=len(x)
  c=x.shape[1]
  A=np.zeros((l,c,c))
  for i in range(l):
    x1=x[i,:,:]
    a=np.corrcoef(x1)
    a=abs(a)
    A[i,:,:]=a
  ai=np.mean(A, axis=0)
  p=np.percentile(ai,80)
  ai[ai<p]=0
  return ai,ai

def ldata(X, y, train_idxs, test_idxs, batch_size, dataset):
    X_train1 = X[train_idxs]
    y_train = y[train_idxs]
    X_test1 = X[test_idxs]
    y_test = y[test_idxs]

    #selected_channels = [0, 8, 10, 12, 16, 17]  # 6
    #selected_channels = [1,2,5,7,8,9,10,11,13,15,17,21,23,24,25,27,28,29]   #18
    selected_channels = [9,10,11,15,16,17,23,27]  #8
    X_train = X_train1[:, selected_channels, :]
    X_test = X_test1[:, selected_channels, :]

    pdc, dtf = nPDC(X_train)
    ap, _ = PCC(X_train)
    ap = torch.tensor(ap, dtype=torch.float).to(device)
    pdc = torch.tensor(pdc, dtype=torch.float).to(device)
    dtf = torch.tensor(dtf, dtype=torch.float).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float, requires_grad=True).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float, requires_grad=True).to(device)
    X_train = torch.tensor(X_train, dtype=torch.float, requires_grad=True).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float, requires_grad=True).to(device)
    full_dataset1 = Data.TensorDataset(X_train, y_train)
    full_dataset2 = Data.TensorDataset(X_test, y_test)
    rtrain_loader = Data.DataLoader(dataset=full_dataset1, batch_size=batch_size, shuffle=False)  # True
    rtest_loader = Data.DataLoader(dataset=full_dataset2, batch_size=batch_size, shuffle=False)  # True

    return rtrain_loader, rtest_loader, ap, pdc, dtf  # ag,

def FL1(epochs, model, train_loader, test_loader, a, loss_func, optimizer):
    total_train_time = 0  # Initialize total training time
    total_test_time = 0  # Initialize total testing time

    for epoch in range(epochs):
        start_train_time = time.time()
        loss, _ = train(model, epoch, train_loader, a, loss_func, optimizer)
        end_train_time = time.time()
        train_time = end_train_time - start_train_time
        total_train_time += train_time  # Accumulate training time
        start_test_time = time.time()
        w, acc0 = evaluate(model, epoch, test_loader, a, loss_func)
        end_test_time = time.time()
        test_time = end_test_time - start_test_time
        total_test_time += test_time  # Accumulate testing time


    return acc0, w,total_train_time,total_test_time

def FLb(epochs, model, train_loader, test_loader, a, loss_func, optimizer):
    total_train_time = 0  # Initialize total training time
    total_test_time = 0  # Initialize total testing time

    for epoch in range(epochs):
        start_train_time = time.time()
        loss, _ = trainb(model, epoch, train_loader, a, loss_func, optimizer)
        end_train_time = time.time()
        train_time = end_train_time - start_train_time
        total_train_time += train_time  # Accumulate training time
        start_test_time = time.time()
        w, acc0 = evaluate(model, epoch, test_loader, a, loss_func)
        end_test_time = time.time()
        test_time = end_test_time - start_test_time
        total_test_time += test_time  # Accumulate testing time

    return acc0, w,total_train_time,total_test_time

device=torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")#
epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
dataset_dir = "D:/DATA/"  + "/"
rnn_suffix = ".mat_dataset.pkl"
rnn_suffix_a = ".mat_A.pkl"

label_list=[".mat_labelV.pkl",".mat_labelA.pkl",".mat_labelD.pkl",".mat_labelL.pkl"]
fold=10
for l in label_list:
    label_suffix = l
    if l==".mat_labelD.pkl" :
        file_list=['s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22', 's23','s24', 's25', 's26', 's28', 's29', 's30','s31','s32' ]
    else:
        file_list=['s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22', 's23','s24', 's25', 's26', 's27', 's28', 's29', 's30','s31','s32' ] #
    for data_file in file_list:
        ###load training set
        print('当前处理的被试：',data_file)
        data = pickle.load(open(dataset_dir + data_file + rnn_suffix, 'rb'), encoding='utf-8')
        labels = pickle.load(open(dataset_dir + data_file + label_suffix, 'rb'), encoding='utf-8')
        labels = np.reshape(labels, (len(labels), -1))
        from sklearn.preprocessing import OneHotEncoder

        en = OneHotEncoder()
        en.fit(labels)
        label = en.transform(labels).toarray()

        mean_acc0 = 0
        mean_acc1 = 0
        acclist0 = []
        acclist1 = []
        fold_tr1_time=0
        fold_te1_time=0
        fold_trb_time=0
        fold_teb_time=0
        for curr_fold in range(fold):

            fold_size = data.shape[0] // fold
            indexes_list = [i for i in range(len(data))]
            indexes = np.array(indexes_list)
            split_list = [i for i in range(curr_fold * fold_size, (curr_fold + 1) * fold_size)]
            split_test = np.array(split_list)
            split_train = np.array(list(set(indexes_list) ^ set(split_list)))
            rtrain_loader, rtest_loader, a0, pdc, dtf = ldata(data, label, split_train, split_test, batch_size,
                                                              dataset=0)
            a1 = torch.unsqueeze(a0, 2)
            pdc1 = torch.unsqueeze(pdc, 2)
            dtf1 = torch.unsqueeze(dtf, 2)
            a = torch.cat((torch.cat((a1, pdc1), dim=2), dtf1), dim=2)

            model0 = gcn1(len(split_train)).to(device)
            model1 = gcn1(len(split_train)).to(device)

            loss_func = torch.nn.CrossEntropyLoss().to(device)
            optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
            acc1, w1,trb,teb = FLb(epochs, model1, rtrain_loader, rtest_loader, a, loss_func, optimizer1)
            fold_trb_time += trb
            fold_teb_time += teb
            mean_acc1 += acc1
        m_acc0 = mean_acc0 / fold
        m_acc1 = mean_acc1 / fold
        print('file{} ,{:.4f},{:.4f}'.format(data_file, m_acc0 * 100, m_acc1 * 100))
        print(f"FLb训练10fold用时：{fold_trb_time:.4f}")
        print(f"FLb测试10fold用时：{fold_teb_time:.4f}")
    print('-------------------------', l)

