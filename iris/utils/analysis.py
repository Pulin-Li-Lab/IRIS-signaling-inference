# helper functions for non-model-related analysis 

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gseapy import prerank
from pandas import DataFrame

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from anndata import AnnData
    from scvi.module import VAE

def get_components(
        trajectory: list[str], 
        subsets: list[NDArray], 
        n_comps: int
    ) -> list[int]:
    '''
    Finds the diffusion components that properly order the given trajectory given AnnData. 

    Args:
        trajectory: list of strings of cell types, in order
        subsets: list of ['X_diffmap'] data corresponding to the cell types in order
        n_comps: number of relevant diffusion components

    Returns:
        components: list of integers of diffusion components that properly order trajectory
    '''
    components = []
    n = len(trajectory)

    for i in range(1, n_comps): # relevant diffusion components start at index 1
        front = True
        back = True
        forward = []
        reverse = []
        for cell_type in trajectory:
            diff_component = subsets[cell_type][:,i]
            avg_resp = np.mean(diff_component)
            # collect values along diffusion component
            forward.append(avg_resp)
            # collect values along reverse of diffusion component
            reverse.append(-1*avg_resp)
        for j in range(1, n):
            # diffusion component incorrectly orders
            if forward[j] < forward[j-1]:
                front = False
            if reverse[j] < reverse[j-1]:
                back = False
        # diffusion component orders correctly in either forward or reverse
        if front or back: 
            components.append(i+1)

    return components

def calc_enrichment(
        vae: VAE, 
        adata: AnnData
    ) -> tuple[DataFrame, DataFrame]:
    '''
    Helper function that calculates enrichment score using GSEApy.

    Args:
        vae: VAE model to extract gradients from
        adata: AnnData object

    Returns:
        pre_res.res2d['NES']: 'NES' column of GSEApy preprank
        pre_res.res2d['Term']: 'Term' column of GSEApy prerank
    '''
    for param in vae.module.parameters():
        x = (param.grad)
        break
        
    arr = np.divide(np.abs(x.cpu().numpy()), 1)
    arr[(arr == np.inf)] = 0
    
    df = DataFrame({'0': adata.var.index,'1':arr})
    df.index = df['0']
    df = df.drop('0', axis=1)
    
    pre_res = prerank(rnk=df, # or rnk = rnk,
                gene_sets='KEGG_2016',
                threads=4,
                min_size=5,
                max_size=1000,
                permutation_num=1000, # reduce number to speed up testing
                outdir=None, # don't write to disk
                seed=6,
                verbose=True, # see what's going on behind the scenes
                )

    # categorical scatterplot
    pre_res.res2d.index = pre_res.res2d['Term']
    
    return pre_res.res2d['NES'], pre_res.res2d['Term']