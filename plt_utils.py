import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd
from matplotlib import cm


def plt_IDerr(ID_err, full_err, k_ls, bsave=False, plt_title='', fname=''):
    fig, ax = plt.subplots()

    ax.plot(k_ls, ID_err, label='ID')
    ax.axhline(y=full_err, color='r', linestyle='--', label='original NN')
    ax.set(xlabel='k', ylabel='test loss', 
            title=plt_title)
    ax.grid()
    ax.legend()

    if bsave:
        fig.savefig(fname)

    plt.show()

def plt_weights(model):
    fullWeights = model.fc1.weight.detach().numpy()
    fullAlf     = model.fc2.weight.detach().numpy()

    fig   = plt.figure(figsize=(10,10))
    ax    = fig.add_subplot(111, projection='3d')
    graph =ax.scatter(fullWeights[:,0],fullWeights[:,1],fullWeights[:,2],
                                            marker='.',c=fullWeights[:,3] )

    plt.show()

def plt_mult(full_loss, full_acc, all_loss, all_acc, 
        names, k_ls, title, 
        bsave, fname, 
        yscale='linear', epsilon=1e-3, 
        xlabel='k', ylabel=['test loss','test acc']):
    fig, axs = plt.subplots(2)
    fig.suptitle(title)

    range_loss = [np.min(all_loss), np.max(all_loss)]
    range_loss = [np.minimum(range_loss[0],full_loss)-epsilon, 
                    np.maximum(range_loss[1],full_loss)+epsilon]
    range_acc  = [np.min(all_acc), np.max(all_acc)]
    range_acc = [np.minimum(range_acc[0],full_acc)-epsilon, 
                    np.maximum(range_acc[1],full_acc)+epsilon]
    for j in range(all_loss.shape[1]):
        #if 'random' in names[j]:
        #    continue
        axs[0].plot(k_ls, all_loss[:,j], label=names[j])
        axs[1].plot(k_ls, all_acc[:,j], label=names[j])


    axs[0].set_ylim(range_loss)
    axs[1].set_ylim(range_acc)
    #axs[0].set_ylim([0.56,0.59])
    #axs[1].set_ylim([0.795,0.81])

    axs[0].axhline(y=full_loss,color='r', linestyle='--', label='original NN')
    axs[1].axhline(y=full_acc,color='r', linestyle='--', label='original NN')

    axs[0].set(xlabel=xlabel, ylabel=ylabel[0])
    axs[1].set(xlabel=xlabel, ylabel=ylabel[1])

    axs[0].set_yscale(yscale)
    axs[1].set_yscale(yscale)

    axs[0].grid()
    axs[1].grid()

    axs[0].legend(bbox_to_anchor=(1.04,1), loc="upper left")
    axs[1].legend(bbox_to_anchor=(1.04,1), loc="upper left")

    plt.subplots_adjust(right=0.65) 
        
    if bsave:
        fig.savefig(fname)

    plt.show()

def lookup(fname):
    i,j = 0,0
    names = gen_names()
    plk = {'id':0, 'neuron-mag':1, 'random':2, 'sparsereg':3}
    nlk = {'none':0, 'last1':1, 'last2':2, '1':3, '2':4, 'var':5}
    klk = {'8':0, '16':1, '32':2, '64':3, '128':4}
    f_ls = fname.split('_')
    if f_ls[1] == 'ft':
        g_ls = f_ls[3].split('lr')
        i = klk[g_ls[0][1:]]
        for p in plk.keys():
            if p in f_ls[2]:
                nstr = f_ls[2].replace(p,'')
                j = plk[p]*5 + nlk[nstr]
                return (i,2*j, 2*j+1)
    else:
        g_ls = f_ls[1].split('lr')
        i = klk[g_ls[0][1:]]
        return(i,len(names)-1)


def gen_names():
    names = []
    pls = ['id', 'neuron-mag', 'random', 'sparsereg']
    nls = ['none', 'last1', 'last2', '1', '2', 'var']
    ftls = ['noft', 'ft']
    for p in pls:
        for n in nls:
            for ft in ftls:
                names.append(p+':'+n+':'+ft)
    names.append('direct train')
    return names

def gen_kls(fdir):
    return 1


def plt_mult2(fdir, full_acc, full_loss, testbool=True):
    names = gen_names()
    k_ls = [8,16,32,64,128]
    all_loss = np.zeros((len(k_ls),len(names)))
    all_acc  = np.zeros((len(k_ls),len(names)))
    if testbool: key='test' 
    else: key='train'
    for fname in os.listdir(fdir):
        if 'ft' in fname:
            state = torch.load(fdir+fname)
            (i,jno,jft) = lookup(fname)
            # just prune
            all_loss[i,jno] = state['loss'][key].iloc[0]
            all_acc[i,jno]  = state['acc'][key].iloc[0]
            # after finetune
            all_loss[i,jft] = state['loss'][key].iloc[-1]
            all_acc[i,jft]  = state['acc'][key].iloc[-1]
        else:
            state = torch.load(fdir+fname)
            (i,j) = lookup(fname)
            all_loss[i,j] = state['loss'][key].iloc[-1]
            all_acc[i,j]  = state['acc'][key].iloc[-1]
            
    plt_mult(full_loss, full_acc, all_loss, all_acc, names, k_ls, title='', bsave=False, fname='', yscale='linear')
    return all_loss, all_acc, names, k_ls


def plt_train(fdir, epochs):
    names = gen_names()
    all_loss = np.zeros()

    #TODO WRITE, have version written in FashionExamine.ipynb

def main():
    full_acc = 0.8892
    full_loss = 0.323121
    plt_mult2('../data/mult1/', full_acc, full_loss, True)

if __name__ == '__main__':
    main()
