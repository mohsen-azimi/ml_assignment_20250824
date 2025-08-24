# -----------------------------------------------------------------------------
# Connectionist Temporal Classification (CTC) — Graves et al., ICML 2006
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.autograd import Function
import q4_utils as utils


class CTCLoss(Function):
    @staticmethod
    def forward(ctx, logits, targets, target_lengths, input_lengths, blank):
        """
        [Sec. 4.2] Eq. (16).
        """
        T, N, K = logits.shape

        # keep these for backward
        ctx.blank = blank
        ctx.save_for_backward(logits, targets, target_lengths, input_lengths)

        # loss
        losses = []
        cursor_idx = 0
        for n in range(N):
            T_n = int(input_lengths[n])  # paper’s T
            U_n = int(target_lengths[n])  # paper’s U

            tc = targets_cat[cursor_idx: cursor_idx + U_n].to(logits.device)
            loss_n, _ = utils.ctc_gradients_wrt_logits(logits[:T_n, n, :], tc, blank)
            losses.append(loss_n)
            cursor_idx += U_n

        return torch.stack(losses).mean() if losses else logits.new_zeros(())  #  mean loss over batch

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backprop:  Eq. (16) gradients and scale
        """
        logits, targets, target_lengths, input_lengths = ctx.saved_tensors
        blank = ctx.blank

        T, N, K = logits.shape
        device = logits.device

        grads = torch.zeros_like(logits)

        #  grad wrt logits via Eq. (16)
        cursor_idx = 0
        for n in range(N):
            T_n = int(input_lengths[n])  # paper’s T
            U_n = int(target_lengths[n])  # paper’s U

            tc = targets[cursor_idx: cursor_idx + U_n].to(device)
            _, g = utils.ctc_gradients_wrt_logits(logits[:T_n, n, :], tc, blank)  # (L,K)
            grads[:T_n, n, :] = g
            cursor_idx += U_n

        # Scale
        grads = grads * (grad_out / max(N, 1)).to(logits.dtype)

        return grads, None, None, None, None, None  # only logits require gradients


# -----------------------------------------------------------------------------
# Bi-LSTM [Sec. 5]
# -----------------------------------------------------------------------------
class Bi_LSTM(nn.Module):
    def __init__(self, input_dim, hidden, num_classes, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, num_layers=num_layers, bidirectional=True)
        self.linear = nn.Linear(2 * hidden, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), enforce_sorted=False)
        y, _ = nn.utils.rnn.pad_packed_sequence(self.rnn(packed)[0])
        return self.linear(y)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(66)

    # Shapes and params
    T, N, D, K = 200, 12, 64, 40  # time, batch, feature dim, classes (incl. blank)
    BLANK = 0

    # lengths inputs
    lengths = torch.tensor([25, 21], dtype=torch.long)

    # Random inputs
    x = torch.randn(T, N, D)

    # some targets (w/o blanks)
    l0 = torch.tensor([1, 2, 3], dtype=torch.long)
    l1 = torch.tensor([2, 4], dtype=torch.long)

    target_lengths = torch.tensor([int(t.numel()) for t in [l0, l1]], dtype=torch.long)
    targets_cat = torch.cat([l0, l1], dim=0) if int(lengths.sum()) > 0 else torch.empty(0, dtype=torch.long)

    # Model > loss > optimizer
    bi_lstm = Bi_LSTM(D, hidden=16, num_classes=K, num_layers=1)

    # Single forward + backward step
    logits = bi_lstm(x, lengths)
    loss = CTCLoss.apply(logits, targets_cat, target_lengths, lengths, BLANK)
    opt = torch.optim.SGD(bi_lstm.parameters(), lr=1e-3, momentum=0.9)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("loss :", float(loss.detach()))

    # Decode a sequence for debug
    n = 0
    logits_n = logits[:lengths[n], n]
    pred_best_path = utils.best_path_decoder(logits_n, blank=BLANK)
    ref_n = [l0.tolist(), l1.tolist()][n]
    ler = utils.label_error_rate([pred_best_path], [ref_n])
    print("n=0:\n -- best_path:", pred_best_path, "\n -- prefix:", "n/a", "\n -- ref:", ref_n, "\n -- LER:", ler)
