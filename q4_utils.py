# -----------------------------------------------------------------------------
# Connectionist Temporal Classification (CTC) — Graves et al., ICML 2006
# -----------------------------------------------------------------------------

import torch
from rapidfuzz.distance import Levenshtein as L  # pip install rapidfuzz


# -----------------------------------------------------------------------------
# [Sec. 2.1] Label Error Rate (LER), Eq. (1)
# -----------------------------------------------------------------------------

def label_error_rate(preds, references):
    """
    [Sec. 2.1] Label Error Rate (LER), Eq. (1)
    """
    edit_distance = L.distance
    ler, total, count = 0.0, 0.0, 0
    for pred, ref in zip(preds, references):
        total += edit_distance(pred, ref) / (len(ref))
        count += 1
    ler = total / count
    return ler


# -----------------------------------------------------------------------------
# [Sec. 3] Decoding: best-path (Eq. 4) and prefix-search (skipped this one for now)
# -----------------------------------------------------------------------------
def many2one(path, blank):
    """
    B(π): remove consecutive duplicates, then blanks
    """
    out = []
    previous = None
    for s in path:
        if s != previous:  # skip duplicates
            if s != blank:  # skip blanks
                out.append(s)
            previous = s  # update previous
    return out


def best_path_decoder(logits, blank):
    """
    [Sec. 3.1/3.2] Eq. (4) best-path (fast, less accurate)
    """
    path = torch.softmax(logits, dim=-1).argmax(dim=-1)
    return many2one(path.tolist(), blank)


def prefix_search_decoding(logits, blank, threshold=0.9999):
    """
    To keep the codebase simple, I skipped this method.
    Using best_path_decoder instead
    """
    return best_path_decoder(logits, blank)



# -----------------------------------------------------------------------------
# [Sec. 4.1] Forward–Backward (alpha/beta) with scaling
# -----------------------------------------------------------------------------
def add_blanks(l, blank):  # l = target
    '''
    Build l' = (blank, l1, blank, l2, ..., blank, lU, blank)
    '''
    device = l.device
    U = l.numel()
    l_prime = torch.empty(2 * U + 1, dtype=torch.long, device=device)
    if U == 0:
        l_prime[-1] = blank
    else:
        l_prime[1::2] = l
        l_prime[0::2] = blank
        l_prime[-1] = blank
    return l_prime


@torch.no_grad()
def ctc_forward_and_backward(probs, l, blank):
    """
    [Sec. 4.1 + Sec. 4.2] alpha/beta with scaling; method = dynamic programming
    """
    T, K = probs.shape
    l_prime = add_blanks(l, blank)  # -> 2L+1
    S = l_prime.numel()
    device = probs.device

    eps = torch.tensor(1e-19, device=device, dtype=probs.dtype)

    alpha = torch.zeros((T, S), device=device, dtype=probs.dtype)
    beta = torch.zeros((T, S), device=device, dtype=probs.dtype)

    C = torch.empty(T, dtype=probs.dtype, device=device)
    D = torch.empty(T, dtype=probs.dtype, device=device)

    # ############# Forward (alpha)  (Page 373) #############
    t = 0  # initial time, t=0, base case for DP
    alpha[t, 0] = probs[t, blank]
    if S > 1:
        alpha[t, 1] = probs[t, int(l_prime[1])]

    C[t] = torch.max(alpha[t].sum(), eps)
    alpha[t] /= C[t]

    # recursion (left to right)
    for t in range(1, T):  # loop forwards
        for s in range(S):
            lab = int(l_prime[s])
            total = alpha[t - 1][s]
            if s - 1 >= 0:
                total += alpha[t - 1][s - 1]
            if s - 2 >= 0 and lab != blank and lab != int(l_prime[s - 2]):
                total += alpha[t - 1][s - 2]
            alpha[t][s] = total * probs[t, lab]
        C[t] = torch.max(alpha[t].sum(), eps)
        alpha[t] /= C[t]

    # ############# Backward (beta)  (Page 373) #############
    # t=T-1 init
    beta[-1, -1] = probs[-1, blank]
    if S - 2 >= 0:
        beta[-1, -2] = probs[-1, int(l_prime[-2])]
    D[-1] = torch.max(beta[-1].sum(), eps)
    beta[-1] /= D[-1]

    # recursion (right to left)
    for t in range(T - 2, -1, -1):  # loop backwards
        for s in range(S):
            l = int(l_prime[s])
            total = beta[t + 1][s]
            if s + 1 < S:
                total += beta[t + 1][s + 1]
            if s + 2 < S and l != blank and l != int(l_prime[s + 2]):
                total += beta[t + 1][s + 2]
            beta[t][s] = total * probs[t, l]
        D[t] = torch.max(beta[t].sum(), eps)
        beta[t] /= D[t]

    p_lx = alpha[-1, -1] + (alpha[-1, -2]) # eqn 8
    log_p = torch.log(p_lx) + torch.log(C).sum()

    return log_p, alpha, beta, C, D, l_prime


# -----------------------------------------------------------------------------
# [Sec. 4.2] Gradient wrt logits
# -----------------------------------------------------------------------------
@torch.no_grad()
def ctc_gradients_wrt_logits(logits, l, blank):
    probs = torch.softmax(logits, dim=-1)
    grad = torch.zeros_like(logits)

    log_p, alpha, beta, C, D, l_prime = ctc_forward_and_backward(probs, l, blank)
    T, K = probs.shape
    S = l_prime.numel()

    # Eqn (16)
    PiProduct = torch.ones(T, dtype=probs.dtype, device=probs.device)
    for t in range(T - 2, -1, -1):
        PiProduct[t] = PiProduct[t + 1] * (D[t + 1] / C[t + 1])
    Q = D * PiProduct  # Eqn (16)

    # Posterior
    alpha_beta = alpha * beta

    for t in range(T):
        # Sum over occurrences in the l (no blanks) ; Eq. (16)
        bigSum = torch.zeros(K, dtype=logits.dtype, device=logits.device)
        for s in range(S):
            k = int(l_prime[s])
            if k == blank: # skip blanks
                continue
            bigSum[k] += alpha_beta[t, s]
        y_t = torch.clamp(probs[t], 1e-19)  # to avoid /0
        grad[t] = probs[t] - (Q[t] / y_t) * bigSum

    return -log_p, grad
