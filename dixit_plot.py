"""
Plot Dixit data perturb-seq results 

"""
import numpy as np
import os
#import networkx as nx
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from realdata.dixit.dixit_meta import DIXIT_ESTIMATED_FOLDER, DIXIT_FIGURES_FOLDER
from realdata.dixit.dixit_meta import nnodes, EFFECTIVE_NODES, true_B_dixit_paper, true_B_igsp_paper

B0 = true_B_dixit_paper.copy()
np.fill_diagonal(B0, 0)
B0_skeleton = B0 + B0.T
B0_skeleton[np.where(B0_skeleton)] = 1

B1 = true_B_igsp_paper.copy()
np.fill_diagonal(B1, 0)
B1_skeleton = B1 + B1.T
B1_skeleton[np.where(B1_skeleton)] = 1

ALGS2COLORS = dict(zip(['ours','utigsp', 'utigsp_star'], sns.color_palette()))
ALGS2MARKERS = {'ours': '*', 'utigsp': 'o', 'utigsp_star': 's'}

def read_results(file, B, skeleton, nodes='all', method='utigsp',delete_bi_directions=False):
    tp = []
    fp = []
    tp_skeleton = []
    fp_skeleton = []
    for vals in file.keys():
        if method == 'utigsp':
            estimated_dag = file[vals]['estimated_dag']
        else:
            estimated_dag = file[vals]['estimated_cpdag']
            if delete_bi_directions == True:
                estimated_dag[np.where(estimated_dag*estimated_dag.T)] = 0
            
        estimated_skeleton = file[vals]['estimated_skeleton']
        if nodes == 'all':
            tp.append(int(np.sum(estimated_dag*B)))
            fp.append(int(np.sum(estimated_dag)-tp[-1]))
            tp_skeleton.append(int(np.sum(estimated_skeleton*skeleton)/2))
            fp_skeleton.append(int(np.sum(estimated_skeleton)/2 - tp_skeleton[-1]))    
        else:
            tp.append(int(np.sum(estimated_dag[:,EFFECTIVE_NODES]*B[:,EFFECTIVE_NODES])))
            fp.append(int(np.sum(estimated_dag[:,EFFECTIVE_NODES])-tp[-1]))
            tp_skeleton.append(int(np.sum(estimated_skeleton[:,EFFECTIVE_NODES]*skeleton[:,EFFECTIVE_NODES]))-\
                               int(np.sum(estimated_skeleton[EFFECTIVE_NODES][:,EFFECTIVE_NODES]*skeleton[EFFECTIVE_NODES][:,EFFECTIVE_NODES])))
            fp_skeleton.append(int(np.sum(estimated_skeleton[:,EFFECTIVE_NODES]) -\
                                   np.sum(estimated_skeleton[EFFECTIVE_NODES][:,EFFECTIVE_NODES]) - tp_skeleton[-1]))                
    
    return tp, fp, tp_skeleton, fp_skeleton

#%%
# 'dixit_paper' or 'igsp_paper' for ground truth reference
reference = 'dixit_paper'
utigsp_ci_test = 'gauss'    

if reference == 'dixit_paper':
    B = B0
    correct_skeleton = B0_skeleton
elif reference == 'igsp_paper':
    B = B1
    correct_skeleton = B1_skeleton

n_possible_skeleton = int(nnodes*(nnodes-1)/2)
n_true_skeleton = int(np.sum(correct_skeleton)/2)

nI = len(EFFECTIVE_NODES)
n_possible_skeleton_int = int(nI*(nI-1)/2 + nI*(nnodes-nI))
n_true_skeleton_int = int(np.sum(correct_skeleton[:,EFFECTIVE_NODES])-np.sum(correct_skeleton[EFFECTIVE_NODES][:,EFFECTIVE_NODES]))

#%%
# load UTIGSP results
with open(DIXIT_ESTIMATED_FOLDER+'/utigsp_gauss.pkl', 'rb') as f:
    res_utigsp_gauss = pickle.load(f)
    
with open(DIXIT_ESTIMATED_FOLDER+'/utigsp_star_gauss.pkl', 'rb') as f:
    res_utigsp_star_gauss = pickle.load(f)   
    
# with open(DIXIT_ESTIMATED_FOLDER+'/res_utigsp_hsic.pkl', 'rb') as f:
#     res_utigsp_hsic = pickle.load(f)
    
# with open(DIXIT_ESTIMATED_FOLDER+'/res_utigsp_star_hsic.pkl', 'rb') as f:
#     res_utigsp_star_hsic = pickle.load(f)    
    
# with open(DIXIT_ESTIMATED_FOLDER+'/res_utigsp_kci.pkl', 'rb') as f:
#     res_utigsp_kci = pickle.load(f)
    
# with open(DIXIT_ESTIMATED_FOLDER+'/res_utigsp_star_kci.pkl', 'rb') as f:
#     res_utigsp_star_kci = pickle.load(f)    
    
# load our results
with open(DIXIT_ESTIMATED_FOLDER+'/our_results.pkl', 'rb') as f:
    res_ours = pickle.load(f)    
    

#%%

utigsp_gauss_tp, utigsp_gauss_fp, utigsp_gauss_tp_skeleton, utigsp_gauss_fp_skeleton = \
    read_results(res_utigsp_gauss, B, correct_skeleton,method='utigsp')

utigsp_star_gauss_tp, utigsp_star_gauss_fp, utigsp_star_gauss_tp_skeleton, utigsp_star_gauss_fp_skeleton = \
    read_results(res_utigsp_star_gauss, B, correct_skeleton,method='utigsp')
 
# utigsp_hsic_tp, utigsp_hsic_fp, utigsp_hsic_tp_skeleton, utigsp_hsic_fp_skeleton = \
#     read_results(res_utigsp_hsic, B, correct_skeleton,method='utigsp')

# utigsp_star_hsic_tp, utigsp_star_hsic_fp, utigsp_star_hsic_tp_skeleton, utigsp_star_hsic_fp_skeleton = \
#     read_results(res_utigsp_star_hsic, B, correct_skeleton,method='utigsp')
    
# utigsp_kci_tp, utigsp_kci_fp, utigsp_kci_tp_skeleton, utigsp_kci_fp_skeleton = \
#     read_results(res_utigsp_kci, B, correct_skeleton,method='utigsp')

# utigsp_star_kci_tp, utigsp_star_kci_fp, utigsp_star_kci_tp_skeleton, utigsp_star_kci_fp_skeleton = \
#     read_results(res_star_utigsp_kci, B, correct_skeleton,method='utigsp')
    
ours_tp, ours_fp, ours_tp_skeleton, ours_fp_skeleton = read_results(res_ours, B, correct_skeleton,method='ours')
        
    
#%%  ======= PLOT ROC for directed edges recovery ==========
plt.clf()
if utigsp_ci_test == 'gauss':
    plt.scatter(utigsp_gauss_fp,utigsp_gauss_tp,label='UTIGSP',marker=ALGS2MARKERS['utigsp'],color=ALGS2COLORS['utigsp'])
    plt.scatter(utigsp_star_gauss_fp,utigsp_star_gauss_tp,label='UTIGSP_star',marker=ALGS2MARKERS['utigsp_star'],color=ALGS2COLORS['utigsp_star'])
elif utigsp_ci_test == 'hsic':
    plt.scatter(utigsp_hsic_fp,utigsp_hsic_tp,label='UTIGSP',marker=ALGS2MARKERS['utigsp'],color=ALGS2COLORS['utigsp'])
    plt.scatter(utigsp_star_hsic_fp,utigsp_star_hsic_tp,label='UTIGSP_star',marker=ALGS2MARKERS['utigsp_star'],color=ALGS2COLORS['utigsp_star'])

plt.scatter(ours_fp,ours_tp,label='ours',marker=ALGS2MARKERS['ours'],color=ALGS2COLORS['ours'])
plt.xlim([0,50])
plt.ylim([0,16])
plt.title('Directed Edges')
plt.xlabel('False positives')
plt.ylabel('True positives')
plt.grid()
plt.legend()
plt.savefig(os.path.join(DIXIT_FIGURES_FOLDER, 'dixit_directed_all_'+utigsp_ci_test+'.eps'))

#%%  ======== PLOT ROC for skeleton recovery ===========

plt.clf()
if utigsp_ci_test == 'gauss':
    plt.scatter(utigsp_gauss_fp_skeleton,utigsp_gauss_tp_skeleton,label='UTIGSP',marker=ALGS2MARKERS['utigsp'],color=ALGS2COLORS['utigsp'])
    plt.scatter(utigsp_star_gauss_fp_skeleton,utigsp_star_gauss_tp_skeleton,label='UTIGSP_star',marker=ALGS2MARKERS['utigsp_star'],color=ALGS2COLORS['utigsp_star'])
elif utigsp_ci_test == 'hsic':
    plt.scatter(utigsp_hsic_fp_skeleton,utigsp_hsic_tp_skeleton,label='UTIGSP',marker=ALGS2MARKERS['utigsp'],color=ALGS2COLORS['utigsp'])
    plt.scatter(utigsp_star_hsic_fp_skeleton,utigsp_star_hsic_tp_skeleton,label='UTIGSP_star',marker=ALGS2MARKERS['utigsp_star'],color=ALGS2COLORS['utigsp_star'])

plt.scatter(ours_fp_skeleton,ours_tp_skeleton,label='ours',marker=ALGS2MARKERS['ours'],color=ALGS2COLORS['ours'])
plt.plot([0, n_possible_skeleton - n_true_skeleton], [0, n_true_skeleton], color='grey')
plt.xlim([0,100])
plt.ylim([0,18])
plt.title('Skeleton')
plt.xlabel('False positives')
plt.ylabel('True positives')
plt.grid()
plt.legend()
plt.savefig(os.path.join(DIXIT_FIGURES_FOLDER, 'dixit_skeleton_all_'+utigsp_ci_test+'.eps'))

