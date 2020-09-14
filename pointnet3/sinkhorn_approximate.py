## this is adapted from 
## https://github.com/gomena/sinkhorn_networks/blob/999fabda4cf6ce319376bc461d34fbfc4e7d5667/sinkhorn_ops.py#L31
## copyright to original authors

import torch
import torch.nn.functional as F

def _log_sum_exp(X, dim, eps=1e-20):
    X = torch.log(torch.exp(X).sum(dim=dim)+eps)
    return X

def _sample_gumbel(shape, eps=1e-20):
    """Samples arbitrary-shaped standard gumbel variables.
    Args:
    shape: list of integers
    eps: float, for numerical stability
    Returns:
    A sample of standard Gumbel random variables
    """
    u = torch.empty(shape).uniform_(0, 1.0)
    return -torch.log(-torch.log(u + eps) + eps)

def sinkhorn(log_alpha, n_iters=20):
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the succesive row and column
    normalization.
    -To ensure positivity, the effective input to sinkhorn has to be
    exp(log_alpha) (elementwise).
    -However, for stability, sinkhorn works in the log-space. It is only at
    return time that entries are exponentiated.
    [1] Sinkhorn, Richard and Knopp, Paul.
    Concerning nonnegative matrices and doubly stochastic
    matrices. Pacific Journal of Mathematics, 1967
    Args:
        log_alpha: 2D tensor (a matrix of shape [N, N])
        or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        n_iters: number of sinkhorn iterations (in practice, as little as 20
        iterations are needed to achieve decent convergence for N~100)
    Returns:
        A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
        converted to 3D tensors with batch_size equals to 1)
    """
    # if len(log_alpha.shape)==2:
    #     n = log_alpha.shape[0]
    #     m = log_alpha.shape[1]
    #     log_alpha = log_alpha.reshape(-1, n, m)
    n = log_alpha.shape[1]
    log_alpha = log_alpha.reshape(-1, n, n)
    for i in range(n_iters):
        # log_alpha -= _log_sum_exp(log_alpha, dim=2)[:, :, None]
        log_alpha -= _log_sum_exp(log_alpha, dim=1)[:, None, :]
        log_alpha -= _log_sum_exp(log_alpha, dim=2)[:, :, None]
    return torch.exp(log_alpha) # become alpha


def gumbel_sinkhorn(log_alpha,
                    temp=1.0, n_samples=1, noise_factor=0, n_iters=20,
                    squeeze=True, normalized=False):
    """Random doubly-stochastic matrices via gumbel noise.
    In the zero-temperature limit sinkhorn(log_alpha/temp) approaches
    a permutation matrix. Therefore, for low temperatures this method can be
    seen as an approximate sampling of permutation matrices, where the
    distribution is parameterized by the matrix log_alpha
    The deterministic case (noise_factor=0) is also interesting: it can be
    shown that lim t->0 sinkhorn(log_alpha/t) = M, where M is a
    permutation matrix, the solution of the
    matching problem M=arg max_M sum_i,j log_alpha_i,j M_i,j.
    Therefore, the deterministic limit case of gumbel_sinkhorn can be seen
    as approximate solving of a matching problem, otherwise solved via the
    Hungarian algorithm.
    Warning: the convergence holds true in the limit case n_iters = infty.
    Unfortunately, in practice n_iter is finite which can lead to numerical
    instabilities, mostly if temp is very low. Those manifest as
    pseudo-convergence or some row-columns to fractional entries (e.g.
    a row having two entries with 0.5, instead of a single 1.0)
    To minimize those effects, try increasing n_iter for decreased temp.
    On the other hand, too-low temperature usually lead to high-variance in
    gradients, so better not choose too low temperatures.
    Args:
        log_alpha: 2D tensor (a matrix of shape [N, N])
        or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        temp: temperature parameter, a float.
        n_samples: number of samples
        noise_factor: scaling factor for the gumbel samples. Mostly to explore
        different degrees of randomness (and the absence of randomness, with
        noise_factor=0)
        n_iters: number of sinkhorn iterations. Should be chosen carefully, in
        inverse corresponde with temp to avoid numerical stabilities.
        squeeze: a boolean, if True and there is a single sample, the output will
        remain being a 3D tensor.
    Returns:
        sink: a 4D tensor of [batch_size, n_samples, N, N] i.e.
        batch_size *n_samples doubly-stochastic matrices. If n_samples = 1 and
        squeeze = True then the output is 3D.
        log_alpha_w_noise: a 4D tensor of [batch_size, n_samples, N, N] of
        noisy samples of log_alpha, divided by the temperature parameter. If
        n_samples = 1 then the output is 3D.
    """
    
    # reshape
    n = log_alpha.shape[1]
    log_alpha = log_alpha.reshape(-1, n, n)
    batch_size = log_alpha.shape[0]

    if normalized:
        log_alpha = torch.log(log_alpha+1e-10)
    else:
        log_alpha = torch.log_softmax(log_alpha/temp, dim=2)
    
    log_alpha = log_alpha.repeat([n_samples, 1, 1]) # log_alpha_w_noise shape = B*S * N*N
    
    if noise_factor > 0:
        noise = _sample_gumbel([n_samples*batch_size, n, n])*noise_factor
        if log_alpha.is_cuda:    
            noise = noise.to(log_alpha.device)
            log_alpha += noise

    sink = sinkhorn(log_alpha, n_iters)
    if n_samples > 1 or squeeze is False:
        sink = sink.reshape([n_samples, batch_size, n, n])
        sink = sink.permute([1, 0, 2, 3])
        log_alpha = log_alpha.reshape([n_samples, batch_size, n, n])
        log_alpha = log_alpha.permute([1, 0, 2, 3])
    # print("sink", sink)
    return sink, log_alpha


if __name__ == '__main__':

    for i in range(1):
        alpha = torch.empty(8, 8).uniform_(-1, 1)
        tau = 0.1
        log_alpha = F.log_softmax(alpha/tau, dim=1)
        S = sinkhorn(log_alpha, n_iters = 40)
        S = S.squeeze(0)
        # print(S)
        print(i)
        print("S*1_N = ", S.sum(dim=0))
        print("S^T*1_N = ", S.sum(dim=1))

