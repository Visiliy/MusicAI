import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from torch import optim
from torch.nn import init
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


class MelodySimDataset(Dataset):
    def __init__(self, dataset, split, transform=None):
        self.data = dataset[split]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        track = self.data[idx]
        audio1_path = track["audio1"]
        audio2_path = track["audio2"]
        label = track["label"]

        waveform1, sr1 = torchaudio.load(audio1_path)
        waveform2, sr2 = torchaudio.load(audio2_path)

        if self.transform:
            waveform1 = self.transform(waveform1)
            waveform2 = self.transform(waveform2)

        return waveform1, waveform2, torch.tensor(label, dtype=torch.float32)


class LinearPerformerAttention(nn.Module):
    def __init__(self, dim, heads=8, feature_dim=256, dropout=0.1, causal=False):
        super().__init__()
        self.heads = heads
        self.feature_dim = feature_dim
        self.head_dim = dim // heads
        self.causal = causal

        self.proj_matrix = nn.Parameter(torch.randn(heads, self.head_dim, feature_dim))
        nn.init.orthogonal_(self.proj_matrix)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def _feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        residual = x
        b, n, d = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        q_proj = torch.einsum('bhnd,hdf->bhnf', q, self.proj_matrix)
        k_proj = torch.einsum('bhnd,hdf->bhnf', k, self.proj_matrix)
        q_proj = self._feature_map(q_proj)
        k_proj = self._feature_map(k_proj)

        if not self.causal:
            k_v = torch.einsum('bhnf,bhnd->bhfd', k_proj, v)
            attention_out = torch.einsum('bhnf,bhfd->bhnd', q_proj, k_v)
            k_proj_sum = k_proj.sum(dim=2)
            z = 1.0 / (torch.einsum('bhnf,bhf->bhn', q_proj, k_proj_sum) + 1e-8)
            attention_out = attention_out * z.unsqueeze(-1)
        else:
            k_cum = k_proj.cumsum(dim=2)
            kv_cum = (k_proj.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(dim=2)
            attention_out = torch.einsum('bhnf,bhnfd->bhnd', q_proj, kv_cum)
            denom = torch.einsum('bhnf,bhnf->bhn', q_proj, k_cum).unsqueeze(-1)
            attention_out = attention_out / (denom + 1e-8)

        attention_out = attention_out.transpose(1, 2).reshape(b, n, -1)
        x = residual + self.to_out(attention_out)
        x = self.norm(x)
        x = x + self.dropout(self.mlp(x))
        x = self.norm(x)
        return x


class LinearPerformerAttentionMusic(nn.Module):
    def __init__(self, dim, heads=8, feature_dim=256, dropout=0.1, causal=False):
        super().__init__()
        self.heads = heads
        self.feature_dim = feature_dim
        self.head_dim = dim // heads
        self.causal = causal

        self.proj_matrix = nn.Parameter(torch.randn(heads, self.head_dim, feature_dim))
        nn.init.orthogonal_(self.proj_matrix)

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        kernel_sizes = [3, 5, 7, 9]
        self.convs = nn.ModuleList([
            nn.Conv1d(self.head_dim, self.head_dim, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def _feature_map(self, x):
        return F.elu(x) + 1

    def _apply_convs(self, x):
        b, h, n, d = x.shape
        x = x.reshape(b * h, n, d).transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2).reshape(b, h, n, d)
        return x

    def forward(self, x, y):
        residual = x
        b, n, d = x.shape
        _, n2, _ = y.shape
        h = self.heads
        q = self.q(x).view(b, n, h, self.head_dim).transpose(1, 2)
        k = self.k(y).view(b, n2, h, self.head_dim).transpose(1, 2)
        v = self.v(y).view(b, n2, h, self.head_dim).transpose(1, 2)
        q_proj = torch.einsum('bhnd,hdf->bhnf', q, self.proj_matrix)
        k_proj = torch.einsum('bhnd,hdf->bhnf', k, self.proj_matrix)
        q_proj = self._feature_map(q_proj)
        k_proj = self._feature_map(k_proj)

        if not self.causal:
            k_v = torch.einsum('bhnf,bhnd->bhfd', k_proj, v)
            k_v = self._apply_convs(k_v)
            attention_out = torch.einsum('bhnf,bhfd->bhnd', q_proj, k_v)
            k_proj_sum = k_proj.sum(dim=2)
            z = 1.0 / (torch.einsum('bhnf,bhf->bhn', q_proj, k_proj_sum) + 1e-8)
            attention_out = attention_out * z.unsqueeze(-1)
        else:
            k_cum = k_proj.cumsum(dim=2)
            kv_cum = (k_proj.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(dim=2)
            attention_out = torch.einsum('bhnf,bhnfd->bhnd', q_proj, kv_cum)
            attention_out = self._apply_convs(attention_out)
            denom = torch.einsum('bhnf,bhnf->bhn', q_proj, k_cum).unsqueeze(-1)
            attention_out = attention_out / (denom + 1e-8)

        attention_out = attention_out.transpose(1, 2).reshape(b, n, -1)
        x = residual + self.to_out(attention_out)
        x = self.norm(x)
        x = x + self.dropout(self.mlp(x))
        x = self.norm(x)
        return x


class AudioToEmbedding(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_mels=128,
                 seq_len=400,
                 embedding_dim=256,
                 max_len=1000):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        self.proj = nn.Linear(n_mels, embedding_dim)
        self.pos_emb = nn.Embedding(max_len, embedding_dim)
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

    def forward(self, waveforms: torch.Tensor):
        batch_size = waveforms.size(0)
        device = waveforms.device
        mels = self.melspec(waveforms)
        mels = torch.log(mels + 1e-9)
        mels = mels.transpose(1, 2)
        if mels.size(1) > self.seq_len:
            mels = mels[:, :self.seq_len, :]
        else:
            pad_len = self.seq_len - mels.size(1)
            mels = torch.cat([mels, torch.zeros(batch_size, pad_len, mels.size(2), device=device)], dim=1)
        x = self.proj(mels)
        positions = torch.arange(0, self.seq_len, device=device).unsqueeze(0)
        pos_embeddings = self.pos_emb(positions)
        x = x + pos_embeddings
        return x


class BaseModel(nn.Module):
    def __init__(self, sample_rate=16000,
                 seq_len=400,
                 embedding_dim=256,
                 heads=8,
                 feature_dim=256,
                 dropout=0.1,
                 causal=False,
                 layers=12):
        super().__init__()
        self.transform_file = AudioToEmbedding(sample_rate=sample_rate, seq_len=seq_len, embedding_dim=embedding_dim)
        self.block1 = nn.ModuleList([LinearPerformerAttention(dim=embedding_dim, heads=heads, feature_dim=feature_dim, dropout=dropout, causal=causal) for _ in range(layers)])
        self.block2 = nn.ModuleList([LinearPerformerAttention(dim=embedding_dim, heads=heads, feature_dim=feature_dim, dropout=dropout, causal=causal) for _ in range(layers)])
        self.music_connect_block = LinearPerformerAttentionMusic(dim=embedding_dim, heads=heads, feature_dim=feature_dim, dropout=dropout, causal=causal)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        self.out = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = self.transform_file(x)
        y = self.transform_file(y)
        for block in self.block1:
            x = block(x)
        for block in self.block2:
            y = block(y)
        out = self.music_connect_block(x, y)
        out = self.ffn(out)
        out = self.out(out)
        return out


def train(model,
          train_loader,
          num_epochs=10,
          lr=1e-4,
          device="mps",
          save_path="best_model.pth"):

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y, labels in train_loader:
            batch_x, batch_y, labels = batch_x.to(device), batch_y.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x, batch_y).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model with train loss {best_loss:.4f}")

    print("🎯 Training finished.")


def validate(model, val_loader, device="cuda" if torch.cuda.is_available() else "cpu", save_path="best_model.pth"):
    model = model.to(device)
    model.load_state_dict(torch.load(save_path))
    model.eval()

    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y, labels in val_loader:
            batch_x, batch_y, labels = batch_x.to(device), batch_y.to(device), labels.to(device).float()
            outputs = model(batch_x, batch_y).squeeze(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * batch_x.size(0)

    val_loss = total_loss / len(val_loader.dataset)
    print(f"📊 Final Validation Loss: {val_loss:.4f}")
    return val_loss


if __name__ == "__main__":
    dataset = load_dataset("amaai-lab/MelodySim")
    transform = torchaudio.transforms.MelSpectrogram()

    train_dataset = MelodySimDataset(dataset, split="train", transform=transform)
    val_dataset = MelodySimDataset(dataset, split="test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = BaseModel(sample_rate=16000, seq_len=1000, embedding_dim=512,
                      heads=8, feature_dim=256, dropout=0.1, causal=False, layers=12)

    train(model, train_loader, num_epochs=30, lr=1e-4, save_path="best_model.pth")

    validate(model, val_loader, save_path="best_model.pth")
