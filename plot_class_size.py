"""
Empirical findings for the max. class size (exponential component of our algo's comp. complexity) and 
size of the changed nodes. 
"""

import numpy as np
from helpers import get_precision_cov
from functions import diff_marginal_noise_sample_direct
import networkx as nx
from networkx.algorithms.clique import find_cliques as find_maximal_cliques
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
from config import SIMULATIONS_FIGURES_FOLDER


def random_dag(p,I_size,density):
    B = np.random.uniform(-1,-0.25,[p,p])* np.random.choice([-1,1],size=[p,p])
    B = np.triu(B)
    np.fill_diagonal(B,0)
    edge_indices = np.triu(np.random.uniform(size=[p,p])<(density/p))
    B = B*edge_indices    
    #G = nx.to_networkx_graph(B,create_using=nx.DiGraph)
    A = np.zeros(B.shape)
    A[np.where(B)] = 1
    #I = list(np.sort(np.random.choice(p,I_size,replace=False)))
    return A

def find_num_paths(A,q=1):
    k = 1
    sum_Ak = np.zeros(A.shape)
    
    b = matrix_power(A,k)
    while np.sum(b) > 0:
        sum_Ak += b*q**k
        k += 1
        b = matrix_power(A,k)
        
    return sum_Ak

def no_I_ancestor_prob(n_repeat,p,I_size,q):
    res = np.zeros(n_repeat)
    A = random_dag(p,I_size,p)
    N = find_num_paths(A,q)
    for _ in range(n_repeat):
        I = list(np.sort(np.random.choice(p,I_size,replace=False)))
        res[_] = np.prod(1-np.clip(N[I][:,I],0,1).flatten())
        
    return res

def find_sizes(p,I_size,density):
    mu = 0.0
    variance = 1.0
    plus_variance = 1.0
    shift = 1.0
    single_threshold = 1e-6
    tol = 1e-9
    B = np.random.uniform(-1,-0.25,[p,p])* np.random.choice([-1,1],size=[p,p])
    B = np.triu(B)
    np.fill_diagonal(B,0)
    edge_indices = np.triu(np.random.uniform(size=[p,p])<(density/p))
    B = B*edge_indices    
    #G = nx.to_networkx_graph(B,create_using=nx.DiGraph)
    # Intervention set
    I = list(np.sort(np.random.choice(p,I_size,replace=False)))
    
    # internal noises will be N(mean,variance) for G1, N(mean+intervention_side*shift,variance) for G2
    mu1 = mu*np.ones(p)
    mu2 = mu*np.ones(p)
    variance1 = variance*np.ones(p)
    variance2 = variance*np.ones(p)
    # shift the noise means
    mu2[I] += shift
    # increase the noise variances
    variance2[I] += plus_variance
    Omega1 = mu1**2 + variance1
    Omega2 = mu2**2 + variance2
    
    B1 = B.copy(); B2 = B.copy()    
    
    G1 = nx.to_networkx_graph(B1,create_using=nx.DiGraph)
    #G2 = nx.to_networkx_graph(B2,create_using=nx.DiGraph)
    
    # now ready to get precision (or generalized precision) matrix
    Theta1, S1 = get_precision_cov(B1, np.diag(Omega1))
    Theta2, S2 = get_precision_cov(B2, np.diag(Omega2))
    Delta_Theta = np.abs(Theta1-Theta2)>tol
    S_Delta = np.where(np.diag(Delta_Theta))[0]   
    
    diff_marginal_noise = diff_marginal_noise_sample_direct(S1, S2)
    # use single_threshold while considering marginal noises
    J0 = np.intersect1d(np.where(np.abs(diff_marginal_noise)<single_threshold)[0],S_Delta)
    S_d = np.setdiff1d(S_Delta,J0)
    J0_anc = np.zeros((p,p))
    
    for idx in range(len(S_d)):
        anc = nx.ancestors(G1,S_d[idx])    
        ancJ0 = anc.intersection(J0)
        J0_anc[list(ancJ0),S_d[idx]] = 1
        
    J0_anc = J0_anc[J0][:,S_d].T
    J0_anc_lists = [J0[np.where(J0_anc[i])[0]] for i in range(len(J0_anc))]
    J0_anc_size = [len(J0_anc_lists[i]) for i in range(len(J0_anc_lists))]
    
    size_argsort_order = np.argsort(J0_anc_size)
    A_mat = np.zeros([len(J0_anc_size),len(J0_anc_size)])
    
    for i in range(len(J0_anc_size)):
        for j in range(len(J0_anc_size)):
            A_mat[i,j] = np.array_equal(J0_anc_lists[size_argsort_order[i]],J0_anc_lists[size_argsort_order[j]])
        
    A_groups = list(find_maximal_cliques(nx.from_numpy_matrix(A_mat)))
    A_groups = [np.sort([size_argsort_order[i] for i in A]) for A in A_groups]
        
    #L = len(A_groups)
    # get the corresponding J0_Al subsets
    #JA_groups = [list(J0_anc_lists[A_groups[i][0]]) for i in range(L)]
    # map to the correct indices in graph
    A_groups = [list(S_d[A_groups[i]]) for i in range(len(A_groups))]
    
    Al_sizes = [len(A) for A in A_groups]
    #print('p_delta:',len(S_Delta),' J0_size:',len(J0), 'I_size:',len(I), ' Al_sizes:',sorted(Al_sizes))
    return len(S_Delta), len(J0), len(I), max(Al_sizes)

def repeat_find_sizes(n_repeat,p,I_size,density):
    p_delta = np.zeros(n_repeat)
    J0_sizes = np.zeros(n_repeat)
    max_Al_sizes = np.zeros(n_repeat)
    for _ in range(n_repeat):
        res = find_sizes(p, I_size, density)
        p_delta[_] = res[0]
        J0_sizes[_] = res[1]
        max_Al_sizes[_] = res[3] 
        
    return p_delta, J0_sizes, max_Al_sizes


#%%
''' for p = 100, see how it changes with density first '''
n_repeat = 1000
p = 100
I_size = 5
#density_list = [1,2,3,4,5,6,7,8,9,10]
density_list = [3,5,10]

p_delta_p100_I5 = np.zeros((len(density_list),n_repeat))
Al_p100_I5 = np.zeros((len(density_list),n_repeat))

for i in range(len(density_list)):
    res_p100_i = repeat_find_sizes(n_repeat=n_repeat,p=p,I_size=I_size,density=density_list[i])
    p_delta_p100_I5[i] = res_p100_i[0]
    Al_p100_I5[i] = res_p100_i[-1]
    print(i)
    
''' for p = 100, see how it changes with density first '''
n_repeat = 1000
p = 100
I_size = 10
#density_list = [1,2,3,4,5,6,7,8,9,10]
density_list = [3,5,10]


p_delta_p100_I10 = np.zeros((len(density_list),n_repeat))
Al_p100_I10 = np.zeros((len(density_list),n_repeat))

for i in range(len(density_list)):
    res_p100_i = repeat_find_sizes(n_repeat=n_repeat,p=p,I_size=I_size,density=density_list[i])
    p_delta_p100_I10[i] = res_p100_i[0]
    Al_p100_I10[i] = res_p100_i[-1]
    print(i)
    

#%%
xticks_size = 12
yticks_size = 12
xlabel_size = 14
ylabel_size = 14
legend_size = 12
legend_loc = 'lower right'
linewidth = 2.5
linestyle = '--'
markersize = 8
markers = ['o','v','P']


#%%
# plot distribution of max class size and S_Delta size.

percentile_loc = (p_delta_p100_I5.shape[-1]*np.arange(0.1,1,0.1)).astype(int)
percentile_ticks = (100*np.arange(0.1,1,0.1)).astype(int)

# for c in pick_densities:
#     h1 = np.histogram(Al_p100_I5[c],bins=10)
#     h2 = np.histogram(p_delta_p100_I5[c],bins=10)
#     plt.plot(h1[1][1:],h1[0],'-o')
#     plt.plot(h2[1][1:],h2[0],'-x')
plt.figure('p=100, |I| = 5')
plt.plot(percentile_ticks,np.sort(Al_p100_I5[0][percentile_loc]),marker=markers[0],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(Al_p100_I5[1][percentile_loc]),marker=markers[1],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(Al_p100_I5[2][percentile_loc]),marker=markers[2],linestyle=linestyle,linewidth=linewidth,markersize=markersize)

plt.plot(percentile_ticks,np.sort(p_delta_p100_I5[0][percentile_loc]),marker=markers[0],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(p_delta_p100_I5[1][percentile_loc]),marker=markers[1],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(p_delta_p100_I5[2][percentile_loc]),marker=markers[2],linestyle=linestyle,linewidth=linewidth,markersize=markersize)

plt.ylim([0,37])
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.xlabel('Percentile',size=xlabel_size)

plt.grid()
plt.legend(['max $|A_l|$ w/ c=3','max $|A_l|$ w/ c=5','max $|A_l|$ w/ c=10','$|S_\Delta|$ w/ c=3', '$|S_\Delta|$ w/ c=5','$|S_\Delta|$ w/ c=10'])
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/p100_I5.eps')


plt.figure('p=100, |I| = 10')
plt.plot(percentile_ticks,np.sort(Al_p100_I10[0][percentile_loc]),marker=markers[0],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(Al_p100_I10[1][percentile_loc]),marker=markers[1],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(Al_p100_I10[2][percentile_loc]),marker=markers[2],linestyle=linestyle,linewidth=linewidth,markersize=markersize)

plt.plot(percentile_ticks,np.sort(p_delta_p100_I10[0][percentile_loc]),marker=markers[0],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(p_delta_p100_I10[1][percentile_loc]),marker=markers[1],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(p_delta_p100_I10[2][percentile_loc]),marker=markers[2],linestyle=linestyle,linewidth=linewidth,markersize=markersize)

plt.ylim([0,70])
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.xlabel('Percentile',size=xlabel_size)

plt.grid()
plt.legend(['max $|A_l|$ w/ c=3','max $|A_l|$ w/ c=5','max $|A_l|$ w/ c=10','$|S_\Delta|$ w/ c=3', '$|S_\Delta|$ w/ c=5','$|S_\Delta|$ w/ c=10'])
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/p100_I10.eps')


#%%
plt.figure('p=100, |I| = 10')

plt.plot(np.sort(p_delta_p100_I10[0]),'-ro',markersize=0.5)
plt.plot(np.sort(p_delta_p100_I10[1]),'-bo',markersize=0.5)
plt.plot(np.sort(p_delta_p100_I10[2]),'-go',markersize=0.5)

plt.plot(np.sort(Al_p100_I10[0]),'-ro',markersize=0.5)
plt.plot(np.sort(Al_p100_I10[1]),'-bo',markersize=0.5)
plt.plot(np.sort(Al_p100_I10[2]),'-go',markersize=0.5)
plt.grid()
plt.legend(['c=3','c=5','c=10'])
#plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/p100_I10.eps')

