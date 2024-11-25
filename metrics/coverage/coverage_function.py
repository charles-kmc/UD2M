import torch
import pandas as pd
import numpy as np
import os
from scipy.stats.mstats import mquantiles


def coverage_eval(x : torch, op: object):
    """
    This function computes the coverage of the posterior mean

    Parameters
    ----------
    x : torch
        True image
    op : object
        Object containing all the necessary information
        
    Returns
    -------
    """
    
    # Distance between samples and posterior mean
    distancelist = [torch.linalg.matrix_norm(sample.detach().cpu().squeeze()-op.post_mean.squeeze(), ord='fro').mean().item() for sample in op.data]

    # Compute quantiles
    quantilelist = mquantiles(distancelist, prob = op.prob, alphap=0.5, betap=0.5)
    # op.quantilelist_list.append(quantilelist)
    
    # Compute the distance between the true image and the posterior mean
    truexdist = torch.linalg.matrix_norm(x.squeeze()-op.post_mean.squeeze(), ord='fro').mean().item() 
    # op.truexdist_list.append(truexdist)
    
    # Check if the true distance is inside the interval
    # op.inside_interval.append((truexdist < quantilelist))
    num_elements_further = sum(np.array(distancelist) > truexdist) / len(distancelist)
    # op.num_dist_greater.append(num_elements_further)

    # Save the results
    temp_df = pd.DataFrame({
        'im_num':list(op.im_num), 
        'quantilelist':list([np.round(quantilelist,4)]),
        'inside_interval':list([(truexdist < quantilelist)]),
        'num_dist_greater':list([np.round(num_elements_further, 4)]),
        'truexdist':[np.round(truexdist, 4)]
    })
    op.save_dir = os.path.join(op.save_dir, f'output_coverage.csv')
    temp_df.to_csv(op.save_dir, mode='a', header=not os.path.exists(op.save_dir))
    
    return {"sampledist":distancelist, "truexdist":truexdist}