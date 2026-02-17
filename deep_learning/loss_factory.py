import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        compress_factor=0.3,
        eps=1e-12,
        lamda_ri=30,
        lamda_mag=70):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.window = torch.hann_window(win_len)
        self.c = compress_factor
        self.eps = eps
        self.lamda_ri = lamda_ri
        self.lamda_mag = lamda_mag

    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        
        device = y_true.device
        
        pred_stft = torch.stft(y_pred, self.n_fft, self.hop_len, self.win_len, self.window.to(device), return_complex=True)
        true_stft = torch.stft(y_true, self.n_fft, self.hop_len, self.win_len, self.window.to(device), return_complex=True)

        pred_mag = torch.abs(pred_stft).clamp(self.eps)
        true_mag = torch.abs(true_stft).clamp(self.eps)
        
        pred_stft_c = pred_stft / pred_mag**(1 - self.c)
        true_stft_c = true_stft / true_mag**(1 - self.c)

        real_loss = torch.mean((pred_stft_c.real - true_stft_c.real)**2)
        imag_loss = torch.mean((pred_stft_c.imag - true_stft_c.imag)**2)
        mag_loss = torch.mean((pred_mag**self.c - true_mag**self.c)**2)

        # SISNR loss
        y_norm = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true),dim=-1,keepdim=True) + 1e-8)
        sisnr = - 2*torch.log10(
            torch.norm(y_norm, dim=-1, keepdim=True) / 
            torch.norm(y_pred - y_norm, dim=-1, keepdim=True).clamp(self.eps) + 
            self.eps
        ).mean()
        
        return self.lamda_ri*(real_loss + imag_loss) + self.lamda_mag*mag_loss + sisnr


class STFTLoss(nn.Module):
    def __init__(self, n_fft=1024, hop_len=120, win_len=600, window="hann_window"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.register_buffer("window", getattr(torch, window)(win_len))

    def loss_spectral_convergence(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

    def loss_log_magnitude(self, x_mag, y_mag):
        return torch.nn.functional.l1_loss(torch.log(y_mag), torch.log(x_mag))

    def forward(self, x, y):
        """x, y: (B, T), in time domain"""
        x = torch.stft(x, self.n_fft, self.hop_len, self.win_len, self.window.to(x.device), return_complex=True)
        y = torch.stft(y, self.n_fft, self.hop_len, self.win_len, self.window.to(x.device), return_complex=True)
        x_mag = torch.abs(x).clamp(1e-8)
        y_mag = torch.abs(y).clamp(1e-8)
        
        sc_loss = self.loss_spectral_convergence(x_mag, y_mag)
        mag_loss = self.loss_log_magnitude(x_mag, y_mag)
        loss = sc_loss + mag_loss

        return loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[240, 120, 50],
        win_lengths=[1200, 600, 240],
        window="hann_window",
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, hs, wl, window)]

    def forward(self, x, y):
        loss = 0.0
        for f in self.stft_losses:
            loss += f(x, y)
        loss /= len(self.stft_losses)
        return loss
    


if __name__=='__main__':
    a = torch.randn(2, 10000)
    b = torch.randn(2, 10000)

    loss_func = HybridLoss()
    loss = loss_func(a, b)
    print(loss)