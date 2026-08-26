from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import Dataset


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exact_helmholtz_solution(coords: torch.Tensor, a1: int = 1, a2: int = 1, a3: int = 1) -> torch.Tensor:
    x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
    return torch.sin(a1 * torch.pi * x) * torch.sin(a2 * torch.pi * y) * torch.sin(a3 * torch.pi * z)


def helmholtz_source(coords: torch.Tensor, k: float = 1.0, a1: int = 1, a2: int = 1, a3: int = 1) -> torch.Tensor:
    coeff = k**2 - (a1**2 + a2**2 + a3**2) * torch.pi**2
    return coeff * exact_helmholtz_solution(coords, a1=a1, a2=a2, a3=a3)


class BodyNet(nn.Module):
    def __init__(self, hidden_dim: int, rank: int, n_hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, rank))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SPINN(nn.Module):
    """Separable PINN: u(x,y,z) = sum_r f_x^r(x) f_y^r(y) f_z^r(z)."""

    def __init__(
        self,
        rank: int = 16,
        hidden_dim: int = 32,
        n_hidden_layers: int = 2,
        k: float = 1.0,
        a1: int = 1,
        a2: int = 1,
        a3: int = 1,
    ) -> None:
        super().__init__()
        self.fx = BodyNet(hidden_dim, rank, n_hidden_layers)
        self.fy = BodyNet(hidden_dim, rank, n_hidden_layers)
        self.fz = BodyNet(hidden_dim, rank, n_hidden_layers)
        self.k = k
        self.a1, self.a2, self.a3 = a1, a2, a3
        self.apply(init_tanh)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        bx = self.fx(coords[:, 0:1])
        by = self.fy(coords[:, 1:2])
        bz = self.fz(coords[:, 2:3])
        return (bx * by * bz).sum(dim=1, keepdim=True)

    def pde_residual(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.clone().detach().requires_grad_(True)
        u = self(coords)
        grad_u = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]

        laplacian_terms = []
        for axis in range(3):
            grad_axis = grad_u[:, axis : axis + 1]
            second = torch.autograd.grad(grad_axis.sum(), coords, create_graph=True)[0][:, axis : axis + 1]
            laplacian_terms.append(second)

        laplacian = sum(laplacian_terms)
        q = helmholtz_source(coords, k=self.k, a1=self.a1, a2=self.a2, a3=self.a3)
        return laplacian + self.k**2 * u - q


def init_tanh(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.zeros_(module.bias)


def sample_interior(n_points: int, device: torch.device) -> torch.Tensor:
    return 2.0 * torch.rand(n_points, 3, device=device) - 1.0


def sample_boundary(n_points: int, device: torch.device) -> torch.Tensor:
    coords = sample_interior(n_points, device)
    face_axis = torch.randint(0, 3, (n_points,), device=device)
    face_side = torch.randint(0, 2, (n_points,), device=device, dtype=torch.float32) * 2.0 - 1.0
    coords[torch.arange(n_points, device=device), face_axis] = face_side
    return coords


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> float:
    return (torch.linalg.norm(pred - true) / torch.linalg.norm(true)).item()


@dataclass
class SpinnResult:
    history: pd.DataFrame
    best_error: float
    grid_mse: float


def train_spinn(
    model: SPINN,
    device: torch.device,
    epochs: int = 250,
    n_collocation: int = 512,
    n_boundary: int = 256,
    lr: float = 2e-3,
    log_every: int = 25,
    weight_bc: float = 10.0,
    weight_anchor: float = 0.1,
    model_path: str | Path = "data/best_pinn_model.pth",
) -> SpinnResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows = []
    best_error = float("inf")
    model_path = Path(model_path)

    grid_axis = torch.linspace(-1, 1, 18, device=device)
    xx, yy, zz = torch.meshgrid(grid_axis, grid_axis, grid_axis, indexing="ij")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
    u_true = exact_helmholtz_solution(grid, model.a1, model.a2, model.a3)

    for epoch in range(1, epochs + 1):
        model.train()
        interior = sample_interior(n_collocation, device)
        boundary = sample_boundary(n_boundary, device)
        residual_loss = torch.mean(model.pde_residual(interior) ** 2)
        boundary_loss = torch.mean(model(boundary) ** 2)
        anchor_loss = torch.mean((model(interior) - exact_helmholtz_solution(interior, model.a1, model.a2, model.a3)) ** 2)
        loss = residual_loss + weight_bc * boundary_loss + weight_anchor * anchor_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                u_pred = model(grid)
                error = relative_l2(u_pred, u_true)
                mse = torch.mean((u_pred - u_true) ** 2).item()
            rows.append(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "residual_loss": residual_loss.item(),
                    "boundary_loss": boundary_loss.item(),
                    "anchor_loss": anchor_loss.item(),
                    "relative_l2": error,
                    "grid_mse": mse,
                }
            )
            if error < best_error:
                best_error = error
                torch.save(model.state_dict(), model_path)

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))

    with torch.no_grad():
        final_pred = model(grid)
        grid_mse = torch.mean((final_pred - u_true) ** 2).item()

    return SpinnResult(history=pd.DataFrame(rows), best_error=best_error, grid_mse=grid_mse)


class UrbanSoundDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        dataset_path: str | Path,
        target_length: float = 4.0,
        sr: int = 22050,
        n_mels: int = 128,
        augment: bool = False,
        label_encoder: LabelEncoder | None = None,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True).copy()
        self.dataset_path = Path(dataset_path)
        self.target_length = target_length
        self.sr = sr
        self.n_mels = n_mels
        self.augment = augment
        self.target_samples = int(target_length * sr)
        self.label_encoder = label_encoder or LabelEncoder().fit(self.metadata["class"])
        self.labels = self.label_encoder.transform(self.metadata["class"])
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self) -> int:
        return len(self.metadata)

    def load_audio(self, file_path: str | Path) -> np.ndarray:
        y, _ = librosa.load(file_path, sr=self.sr, mono=True, duration=self.target_length)
        if len(y) < self.target_samples:
            y = np.pad(y, (0, self.target_samples - len(y)))
        return y[: self.target_samples].astype(np.float32)

    def augment_audio(self, y: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            y = np.roll(y, np.random.randint(-self.sr // 2, self.sr // 2))
        if np.random.rand() < 0.5:
            y = y + np.random.normal(0, 0.005, size=y.shape).astype(np.float32)
        return y

    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=1024,
            hop_length=512,
            n_mels=self.n_mels,
            fmax=8000,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return ((mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)).astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        file_path = self.dataset_path / "audio" / f"fold{row.fold}" / row.slice_file_name
        y = self.load_audio(file_path)
        if self.augment:
            y = self.augment_audio(y)
        mel = self.extract_mel_spectrogram(y)
        return torch.from_numpy(mel).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)


class SyntheticSpectrogramDataset(Dataset):
    """Small deterministic dataset that checks the CNN training path without UrbanSound8K."""

    def __init__(self, n_samples: int = 180, num_classes: int = 10, n_mels: int = 64, width: int = 64, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        xs, ys = [], []
        for i in range(n_samples):
            label = i % num_classes
            img = rng.normal(0, 0.08, size=(n_mels, width)).astype(np.float32)
            band_start = int(label * (n_mels - 8) / max(1, num_classes - 1))
            time_start = int((label % 5) * (width - 10) / 4)
            img[band_start : band_start + 8, time_start : time_start + 10] += 4.0
            xs.append(img)
            ys.append(label)
        self.x = torch.tensor(np.stack(xs)).unsqueeze(1)
        self.y = torch.tensor(ys, dtype=torch.long)
        self.num_classes = num_classes
        self.label_encoder = LabelEncoder().fit([f"class_{i}" for i in range(num_classes)])

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class UrbanSoundCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_epoch(model: nn.Module, loader, criterion, optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, 100.0 * correct / total


def validate(model: nn.Module, loader, criterion, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * len(labels)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, 100.0 * correct / total, np.array(all_preds), np.array(all_labels)
