from abc import ABC, abstractmethod
import torch


def iou_zero_loops(
    masks1: torch.Tensor, masks2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Zero for‑loops. masks1: (M,H,W), masks2: (N,H,W).
    Returns IoU: (M,N).
    """
    M, H, W = masks1.shape
    N = masks2.shape[0]
    # reshape for broadcasting
    m1 = masks1.view(M, 1, H, W).float()
    m2 = masks2.view(1, N, H, W).float()
    # intersection & union
    inter = (m1 * m2).sum(dim=(2, 3))  # (M,N)
    area1 = m1.sum(dim=(2, 3))  # (M,1)
    area2 = m2.sum(dim=(2, 3))  # (1,N)
    union = area1 + area2 - inter  # (M,N)
    return inter / (union + eps)


def iou_one_loop(
    masks1: torch.Tensor, masks2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    One for‑loop over masks1. Vectorized over masks2.
    """
    M, H, W = masks1.shape
    N = masks2.shape[0]
    iou = torch.zeros((M, N), dtype=torch.float, device=masks1.device)
    area2 = masks2.view(N, -1).sum(dim=1)  # (N,)
    for i in range(M):
        m = masks1[i].float().view(1, H, W)  # (1,H,W)
        inter = (m * masks2).view(N, -1).sum(dim=1)  # (N,)
        area1 = m.sum()
        union = area1 + area2 - inter  # (N,)
        iou[i] = inter / (union + eps)
    return iou


def iou_two_loops(
    masks1: torch.Tensor, masks2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Two for‑loops over masks1 and masks2. Lowest memory overhead.
    """
    M, H, W = masks1.shape
    N = masks2.shape[0]
    iou = torch.zeros((M, N), dtype=torch.float, device=masks1.device)
    for i in range(M):
        m = masks1[i].float()
        area1 = m.sum()
        for j in range(N):
            p = masks2[j].float()
            inter = (m * p).sum()
            area2 = p.sum()
            union = area1 + area2 - inter
            iou[i, j] = inter / (union + eps)
    return iou


def incremental_mIoU(M, P_o, IoU):
    """
    M      : list of ground‐truth masks, |M| = M_n
    P_o    : ordered list of predicted masks, length N_max
    IoU    : 2D array of shape (M_n, P_n) with IoU[m][p]
    returns: list mIoU[1..N_max]
    """
    M_n = len(M)
    N_max = len(P_o)
    best_iou = [0.0] * M_n  # best_iou[m] = max IoU seen so far for gt‐mask m
    sum_best = 0.0  # sum of best_iou over all m
    mIoUs = [0.0] * N_max

    for N in range(1, N_max + 1):
        p_idx = P_o[N - 1]  # the Nth prediction
        # update each ground‐truth mask’s best IoU
        for m in range(M_n):
            new_iou = IoU[m][p_idx]
            if new_iou > best_iou[m]:
                sum_best += new_iou - best_iou[m]
                best_iou[m] = new_iou
        # compute mIoU for first N predictions
        mIoUs[N - 1] = sum_best / M_n

    return mIoUs


def oracle_sequence(M, P, IoU):
    """
    M   : ground‐truth list, |M|
    P   : list of all predictions, |P|
    IoU : 2D array IoU[m][p]
    returns: P_o* (list of p‐indices), and mIoU‐curve[1..|P|]
    """
    M_n = len(M)
    P_set = set(P)
    best_iou = [0.0] * M_n
    seq = []
    mIoUs = []
    sum_best = 0.0

    while P_set:
        # find p with max marginal gain
        best_p, best_gain = None, -1.0
        for p in P_set:
            gain = 0.0
            for m in range(M_n):
                delta = IoU[m][p] - best_iou[m]
                if delta > 0:
                    gain += delta
            if gain > best_gain:
                best_gain, best_p = gain, p

        # pick it
        P_set.remove(best_p)
        seq.append(best_p)
        # update best_iou & sum_best
        for m in range(M_n):
            if IoU[m][best_p] > best_iou[m]:
                sum_best += IoU[m][best_p] - best_iou[m]
                best_iou[m] = IoU[m][best_p]

        # record mIoU after adding best_p
        mIoUs.append(sum_best / M_n)

    return seq, mIoUs
