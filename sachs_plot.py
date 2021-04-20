"""
Plot Sachs protein signaling results here
"""
import numpy as np
import os
#import networkx as nx
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from realdata.sachs.sachs_meta import SACHS_ESTIMATED_FOLDER, SACHS_FIGURES_FOLDER, nnodes
from realdata.sachs.sachs_meta import true_dag_old as true_dag_old
from realdata.sachs.sachs_meta import true_dag_recent as true_dag_recent

reference = 'NessSachs2016'
utigsp_ci_test = 'gauss'

ALGS2COLORS = dict(zip(['ours','utigsp', 'utigsp_star'], sns.color_palette()))
ALGS2MARKERS = {'ours': '*', 'utigsp': 'o', 'utigsp_star': 's',}

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


B0 = true_dag_recent.copy()
np.fill_diagonal(B0, 0)
B0_skeleton = B0 + B0.T
B0_skeleton[np.where(B0_skeleton)] = 1

B1 = true_dag_old.copy()
np.fill_diagonal(B1, 0)
B1_skeleton = B1 + B1.T
B1_skeleton[np.where(B1_skeleton)] = 1


if reference == 'NessSachs2016':
    B = B0
    correct_skeleton = B0_skeleton
elif reference == 'Sachs2005':
    B = B1
    correct_skeleton = B1_skeleton

n_possible_skeleton = int(nnodes*(nnodes-1)/2)
n_true_skeleton = int(np.sum(correct_skeleton)/2)

#%%
# load UTIGSP results
with open(SACHS_ESTIMATED_FOLDER+'/utigsp_gauss.pkl', 'rb') as f:
    res_utigsp_gauss = pickle.load(f)
    
with open(SACHS_ESTIMATED_FOLDER+'/utigsp_star_gauss.pkl', 'rb') as f:
    res_utigsp_star_gauss = pickle.load(f)   
    
# with open(SACHS_ESTIMATED_FOLDER+'/res_utigsp_hsic.pkl', 'rb') as f:
#     res_utigsp_hsic = pickle.load(f)
    
# with open(SACHS_ESTIMATED_FOLDER+'/res_utigsp_star_hsic.pkl', 'rb') as f:
#     res_utigsp_star_hsic = pickle.load(f)    
    
# with open(SACHS_ESTIMATED_FOLDER+'/res_utigsp_kci.pkl', 'rb') as f:
#     res_utigsp_kci = pickle.load(f)
    
# with open(SACHS_ESTIMATED_FOLDER+'/res_utigsp_star_kci.pkl', 'rb') as f:
#     res_utigsp_star_kci = pickle.load(f)    
    
# load our results
with open(SACHS_ESTIMATED_FOLDER+'/our_results.pkl', 'rb') as f:
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
        
    
#%%
#  ======= PLOT ROC for directed edges recovery ==========
plt.clf()
if utigsp_ci_test == 'gauss':
    plt.scatter(utigsp_gauss_fp,utigsp_gauss_tp,label='UTIGSP',marker=ALGS2MARKERS['utigsp'],color=ALGS2COLORS['utigsp'])
    plt.scatter(utigsp_star_gauss_fp,utigsp_star_gauss_tp,label='UTIGSP_star',marker=ALGS2MARKERS['utigsp_star'],color=ALGS2COLORS['utigsp_star'])
elif utigsp_ci_test == 'hsic':
    plt.scatter(utigsp_hsic_fp,utigsp_hsic_tp,label='UTIGSP',marker=ALGS2MARKERS['utigsp'],color=ALGS2COLORS['utigsp'])
    plt.scatter(utigsp_star_hsic_fp,utigsp_star_hsic_tp,label='UTIGSP_star',marker=ALGS2MARKERS['utigsp_star'],color=ALGS2COLORS['utigsp_star'])

plt.scatter(ours_fp,ours_tp,label='ours',marker=ALGS2MARKERS['ours'],color=ALGS2COLORS['ours'])

plt.xlim([0,40])
plt.ylim([0,11])
plt.title('Directed Edges')
plt.xlabel('False positives')
plt.ylabel('True positives')
plt.grid()
plt.legend()
plt.savefig(os.path.join(SACHS_FIGURES_FOLDER, 'sachs_directed_roc.eps'))

#  ======== PLOT ROC for skeleton recovery ===========
plt.clf()
if utigsp_ci_test == 'gauss':
    plt.scatter(utigsp_gauss_fp_skeleton,utigsp_gauss_tp_skeleton,label='UTIGSP',marker=ALGS2MARKERS['utigsp'],color=ALGS2COLORS['utigsp'])
    plt.scatter(utigsp_star_gauss_fp_skeleton,utigsp_star_gauss_tp_skeleton,label='UTIGSP_star',marker=ALGS2MARKERS['utigsp_star'],color=ALGS2COLORS['utigsp_star'])
elif utigsp_ci_test == 'hsic':
    plt.scatter(utigsp_hsic_fp_skeleton,utigsp_hsic_tp_skeleton,label='UTIGSP',marker=ALGS2MARKERS['utigsp'],color=ALGS2COLORS['utigsp'])
    plt.scatter(utigsp_star_hsic_fp_skeleton,utigsp_star_hsic_tp_skeleton,label='UTIGSP_star',marker=ALGS2MARKERS['utigsp_star'],color=ALGS2COLORS['utigsp_star'])

plt.scatter(ours_fp_skeleton,ours_tp_skeleton,label='ours',marker=ALGS2MARKERS['ours'],color=ALGS2COLORS['ours'])
plt.plot([0, n_possible_skeleton - n_true_skeleton], [0, n_true_skeleton], color='grey')
plt.title('Skeleton')
plt.xlabel('False positives')
plt.ylabel('True positives')
plt.grid()
plt.legend()
plt.savefig(os.path.join(SACHS_FIGURES_FOLDER, 'sachs_skeleton_roc.eps'))
