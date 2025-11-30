
import cvxpy as cp
import scipy.io
import torch
import numpy as np
import torch.utils.data as Data
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def nPDC(x):
    n = x.shape[0]
    c = x.shape[1]
    b1 = np.zeros((n, c, c))
    c1 = np.zeros((n, c, c))
    for i in range(n):
        x1 = x[i]
        b, c = PDC(x1, 40)
        b1[i] = b
        c1[i] = c
    mb = np.mean(b1, axis=0)
    mc = np.mean(c1, axis=0)
    p1 = np.percentile(mb, 80)
    mb[mb < p1] = 0
    p2 = np.percentile(mc, 80)
    mc[mc < p2] = 0
    return mb, mc

def PCC(x):
    l = len(x)
    c = x.shape[1]
    A = np.zeros((l, c, c))
    for i in range(l):
        x1 = x[i, :, :]
        a = np.corrcoef(x1)
        a = abs(a)
        A[i, :, :] = a
    ai = np.mean(A, axis=0)
    p = np.percentile(ai, 80)
    ai[ai < p] = 0
    return ai, ai


def eucli(A, B):
    return np.sqrt(sum(np.power((A - B), 2)))


def zk(data):
    k = len(data)
    z = []
    for i in range(k - 1):
        for j in range(i + 1, k):
            a1 = data[i]
            a2 = data[j]
            l = eucli(a1, a2)
            z = np.append(z, l)
    return z


def tri(A):
    m = len(A)
    n = int((1 + np.sqrt(1 + 8 * m)) / 2)
    AA = np.zeros((n, n))
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            AA[i, j] = A[index]
            AA[j, i] = A[index]
            index = index + 1
    return AA


def GSL1(X_train):
    l = len(X_train)
    n = 14
    m = int(n * (n - 1) / 2)
    z0 = np.zeros((l, m))
    for k in range(l):
        d = X_train[k]
        z1 = zk(d)
        z0[k] = z1
    z = np.mean(z0, axis=0)

    e = scipy.io.loadmat('D:/DATA/ele.mat')['e14']
    ze = zk(e)

    alpha = 0.9
    beta = 0.1  # 0.01
    one = np.ones((1, n))
    M = scipy.io.loadmat('D:/DATA/fixM0.mat')['M']
    w = cp.Variable(m)
    cost = 2 * (w.T @ z) - alpha * (one @ (cp.log(M @ w))) + 2 * beta * cp.sum_squares(w)
    prob = cp.Problem(cp.Minimize(cost), [w >= 1e-4])
    prob.solve()
    W1 = abs(w.value)
    W = tri(W1)
    i = np.identity(n)
    W = W + i
    p = np.percentile(W, 80)
    W[W < p] = 0
    return W


def GSLn(X_train):
    l = len(X_train)
    n = 14
    m = int(n * (n - 1) / 2)
    e = scipy.io.loadmat('D:/DATA/ele.mat')['e14']
    ze = zk(e)
    alpha = 0.9
    beta = 0.1  # 0.01
    one = np.ones((1, n))
    M = scipy.io.loadmat('D:/DATA/fixM0.mat')['M']
    W = np.zeros((l, n, n))
    for k in range(l):
        d = X_train[k]
        z = zk(d)
        w = cp.Variable(m)
        cost = 2 * (w.T @ z) - alpha * (one @ (cp.log(M @ w))) + 2 * beta * cp.sum_squares(w)  # +2*1*(w.T @ ze)
        prob = cp.Problem(cp.Minimize(cost), [w >= 1e-4])
        prob.solve(solver=cp.SCS)
        w1 = abs(w.value)
        w1 = w1 / max(w1)
        w1 = tri(w1)
        i = np.identity(n)
        w1 = w1 + i
        p = np.percentile(w1, 90)
        w1[w1 < p] = 0
        W[k] = w1
    mw = np.mean(W, axis=0)
    p = np.percentile(mw, 80)
    mw[mw < p] = 0
    return mw


def ldata(X, y, train_idxs, test_idxs, batch_size):
    X_train = X[train_idxs]
    y_train = y[train_idxs]
    X_test = X[test_idxs]
    y_test = y[test_idxs]
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
    rtrain_loader = Data.DataLoader(dataset=full_dataset1, batch_size=batch_size, shuffle=False)
    rtest_loader = Data.DataLoader(dataset=full_dataset2, batch_size=batch_size, shuffle=False)


    return rtrain_loader, rtest_loader, ap, pdc, dtf
