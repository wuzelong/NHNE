import heapq

import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def get_Pk(Ek_unsort, Eq, K):
    n = len(Ek_unsort)
    nk = len(K)
    max_k = K[nk - 1]
    Ek = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if len(Ek) < max_k:
                heapq.heappush(Ek, (Ek_unsort[i][j], (i, j)))
            else:
                heapq.heappushpop(Ek, (Ek_unsort[i][j], (i, j)))
    Pk = []
    cnt = 0.
    list = []
    for i in range(max_k):
        k, v = heapq.heappop(Ek)
        list.append((k, v))
    list.reverse()
    for j in range(1, max_k + 1):
        val, idx = list[j - 1]
        if val > 0. and Eq[idx[0]][idx[1]] == 1:
            cnt += 1
        if j in K:
            Pk.append(round(cnt / j, 4))
    for j in range(nk):
        print(Pk[j], end=' ')


def get_MAP(Ek_unsort, Eq):
    n = len(Ek_unsort)
    MAP = 0
    for i in range(n):
        # sort Ek
        Ek = {}
        for j in range(n):
            Ek[j] = Ek_unsort[i][j]
        Ek = sorted(Ek.items(), key=lambda x: x[1], reverse=True)
        Ek = dict(Ek)
        Pki = 0
        APi = 0
        Delta = 0
        idx = 1
        for key, value in Ek.items():
            if Eq[i][key] == 1:
                if value > 0:
                    Pki += 1
                Delta += 1
                APi += Pki / idx  # precision@k(i) = Pki / j
            idx += 1
        if Delta != 0:
            MAP += APi / Delta  # AP(i) = APi / Delta
    print(round(MAP / n, 4))


def get_EkEq(decode, observe, adj):
    n = len(adj)
    Ek = np.zeros((n, n))
    Eq = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if observe[i][j] == 0:
                Ek[i][j] = decode[i][j]
                Eq[i][j] = adj[i][j]
    return Ek, Eq


def predict_with_n(y_test, y_pred_pro):
    y_pred_pro = np.array(y_pred_pro)
    y_pred_pro = y_pred_pro[:, :, 1].reshape(len(y_pred_pro), len(y_pred_pro[0]))
    y_pred_pro = np.transpose(y_pred_pro)
    y_pred = np.zeros(y_pred_pro.shape)
    for i in range(y_test.shape[0]):
        n = sum(y_test[i])
        top_n = y_pred_pro[i, :].argsort()[-n:]
        y_pred[i, top_n] = 1
    return y_pred


def ten_fold_cross_validation(x, y, seed):
    F1 = np.zeros(2)
    mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    LR = MultiOutputClassifier(
        LogisticRegression(max_iter=1000, penalty='l2', solver='liblinear', random_state=20230317))

    for train_index, test_index in mskf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        LR.fit(x_train, y_train)
        LR_pred = LR.predict_proba(x_test)
        F1[0] += f1_score(y_test, predict_with_n(y_test, LR_pred), average='micro', zero_division=1)
        F1[1] += f1_score(y_test, predict_with_n(y_test, LR_pred), average='macro', zero_division=1)
    return round(F1[0] / 10, 4), round(F1[1] / 10, 4)


def F1_with_different_train_radio(x, y, seed):
    F1 = np.zeros(2, 10)
    mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    LR = MultiOutputClassifier(
        LogisticRegression(max_iter=1000, penalty='l2', solver='liblinear', random_state=20230317))
    idx = []

    for train_index, test_index in mskf.split(x, y):
        idx.append(test_index)

    for i in range(9):
        test_index = []
        train_index = []
        for j in range(i + 1):
            train_index.append(idx[j])
        for j in range(i + 1, 10):
            test_index.append(idx[j])
        test_index = np.concatenate(test_index).ravel()
        train_index = np.concatenate(train_index).ravel()

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        LR.fit(x_train, y_train)
        LR_pred = LR.predict_proba(x_test)
        F1[0, i] = round(f1_score(y_test, predict_with_n(y_test, LR_pred), average='micro', zero_division=1), 4)
        F1[1, i] = round(f1_score(y_test, predict_with_n(y_test, LR_pred), average='macro', zero_division=1), 4)
    return F1
