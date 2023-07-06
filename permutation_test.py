from sklearn.utils.validation import check_random_state
import numpy as np

# 1-sample Permutation test
def permtest_1samp(sample, popmean, axis=0, n_perm=5000, seed=0):
    '''
    One-sample permutation test
    '''
    rs = check_random_state(seed)
    
    # center `sample` around `popmean` and calculate original mean
    zeroed = sample - popmean
    true_mean = zeroed.mean(axis=axis)
    abs_mean = np.abs(true_mean)

    # this for loop is not _the fastest_ but is memory efficient
    # the broadcasting alt. would mean storing zeroed.size * n_perm in memory
    permutations = np.ones(true_mean.shape)
    for perm in range(n_perm):
        flipped = zeroed * rs.choice([-1, 1], size=zeroed.shape)  # sign flip
        permutations += np.abs(flipped.mean(axis=axis)) >= abs_mean   
    p_val = np.sum(permutations) / (n_perm + 1)  # + 1 in denom accounts for true_mean
    
    true_mean_out = str("Mean difference: " + str(true_mean))
    p_val_out = str("p value: " + str(p_val))
    return true_mean_out, p_val_out

# 2-sample Permutation test
def permtest_2samp(a, b, axis=0, n_perm=5000, seed=0):
    '''
    Two-sample permutation test
    '''
    rs = check_random_state(seed)
    # calculate original difference in means
    ab = np.stack([a, b], axis=0)
    if ab.ndim < 3:
        ab = np.expand_dims(ab, axis=-1)
    true_diff = np.squeeze(np.diff(ab, axis=0)).mean(axis=axis) / 1
    abs_true = np.abs(true_diff)

    # idx array
    reidx = np.meshgrid(*[range(f) for f in ab.shape], indexing='ij')

    permutations = np.ones(true_diff.shape)
    for perm in range(n_perm):
        # use this to re-index (i.e., swap along) the first axis of `ab`
        swap = rs.random_sample(ab.shape[:-1]).argsort(axis=axis)
        reidx[0] = np.repeat(swap[..., np.newaxis], ab.shape[-1], axis=-1)
        # recompute difference between `a` and `b` (i.e., first axis of `ab`)
        pdiff = np.squeeze(np.diff(ab[tuple(reidx)], axis=0)).mean(axis=axis)
        permutations += np.abs(pdiff) >= abs_true

    p_val = permutations / (n_perm + 1)  # + 1 in denom accounts for true_diff

    true_diff_out = str("Mean difference: " + str(true_diff))
    p_val_out = str("p value: " + str(p_val))
    return true_diff_out, p_val_out