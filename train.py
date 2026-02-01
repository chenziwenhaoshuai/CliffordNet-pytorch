import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from timm.models.layers import DropPath, trunc_normal_



class CliffordInteraction(nn.Module):


    def __init__(self, dim, shifts=[1, 2]):
        super().__init__()
        self.dim = dim
        self.shifts = shifts


        self.ctx_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )


        self.det_proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.norm_ctx = nn.GroupNorm(1, dim, eps=1e-6)
        self.norm_det = nn.GroupNorm(1, dim, eps=1e-6)

        input_proj_dim = 2 * len(shifts) * dim
        self.final_proj = nn.Conv2d(input_proj_dim, dim, kernel_size=1)

    def forward(self, x):
        """
        Input: x (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 1. Dual-Stream Generation
        # Context Stream (C)
        z_ctx = self.ctx_conv(x)
        # State Stream (H)
        z_det = self.det_proj(x)

        z_ctx = self.norm_ctx(z_ctx)
        z_det = self.norm_det(z_det)
        f_list = []

        # 2. Sparse Rolling Interaction (Algorithm 1, Lines 8-15)
        for s in self.shifts:
            # Cyclic Shift (Translation Operator Ts)

            z_det_s = torch.roll(z_det, shifts=s, dims=1)
            z_ctx_s = torch.roll(z_ctx, shifts=s, dims=1)

            # Geometric Product Components
            # Scalar Component (Dot): eq 7 top -> SiLU(H * Ts(C))

            dot_s = F.silu(z_det * z_ctx_s)

            # Bivector Component (Wedge): eq 7 bottom -> H * Ts(C) - Ts(H) * C
            wedge_s = z_det * z_ctx_s - z_det_s * z_ctx

            f_list.append(dot_s)
            f_list.append(wedge_s)

        # 3. Concatenate & Project (Algorithm 1, Lines 16-17)
        g_raw = torch.cat(f_list, dim=1)  # (B, 2*|S|*C, H, W)
        g_feat = self.final_proj(g_raw)  # (B, C, H, W)

        return g_feat




class CliffordBlock(nn.Module):


    def __init__(self, dim, shifts, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()

        # Input Norm
        self.norm = nn.GroupNorm(1, dim)

        # Geometric Interaction
        self.interaction = CliffordInteraction(dim, shifts=shifts)

        # Gated Geometric Residual (GGR) components
        # Gate Linear: projects cat(x, g_feat) -> dim
        self.gate_linear = nn.Conv2d(dim * 2, dim, kernel_size=1)

        # Layer Scale & DropPath
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: (B, C, H, W)
        shortcut = x

        # 1. Input Norm
        x_ln = self.norm(x)

        # 2. & 3. Efficient Interaction (Returns G_feat)
        g_feat = self.interaction(x_ln)

        # 4. Gated Geometric Residual (Algorithm 1, Lines 19-22)
        # M = Cat([X_ln, G_feat])
        m = torch.cat([x_ln, g_feat], dim=1)

        # alpha = Sigmoid(Linear_gate(M))
        alpha = torch.sigmoid(self.gate_linear(m))

        # H_mix = SiLU(X_ln) + alpha * G_feat
        h_mix = F.silu(x_ln) + alpha * g_feat

        # 5. Output Update (Algorithm 1, Line 25)
        # X = X_prev + Drop(gamma * H_mix)
        x = shortcut + self.drop_path(self.gamma * h_mix)

        return x




class CliffordNet(nn.Module):
    def __init__(self,
                 img_size=32,
                 in_chans=3,
                 num_classes=100,
                 embed_dim=192,
                 depth=12,
                 shifts=[1, 2],
                 drop_path_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Stem: Patch Embedding
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU()
        )

        # Backbone: Stack of Clifford Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            CliffordBlock(
                dim=embed_dim,
                shifts=shifts,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])

        # Head
        self.norm = nn.GroupNorm(1, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Blocks
        for block in self.blocks:
            x = block(x)

        # Head
        x = self.norm(x)
        x = x.mean(dim=[-2, -1])  # Global Average Pooling
        x = self.head(x)
        return x




def cliffordnet_nano(num_classes=100):
    # Nano: ~1.4M params. 2 shifts.
    return CliffordNet(
        img_size=32,
        embed_dim=128,
        depth=12,
        shifts=[1, 2],
        num_classes=num_classes,
        drop_path_rate=0.05
    )


def cliffordnet_fast(num_classes=100):
    # Fast: ~2.6M params. 5 shifts.
    return CliffordNet(
        img_size=32,
        embed_dim=160,
        depth=12,
        shifts=[1, 2, 4, 8, 15],
        num_classes=num_classes,
        drop_path_rate=0.1
    )



def train_cifar100():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BATCH_SIZE = 128
    EPOCHS = 200
    LR = 1e-3
    WD = 0.05

    # 保持 Transforms 不变...
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.1)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])


    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                              pin_memory=True)

    model = cliffordnet_nano(num_classes=100).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)


    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup_epochs = 5
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_epochs),  # 0 -> 5 epoch 预热
        CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6)
    ], milestones=[warmup_epochs])

    criterion = nn.CrossEntropyLoss()

    print("Start Training (Stable Version)...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 检查是否有 NaN
            if torch.isnan(loss):
                print("Error: Loss is NaN! Stopping training.")
                return

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total

        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total


        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{EPOCHS}] "
              f"Loss: {running_loss / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"LR: {current_lr:.6f}")


if __name__ == "__main__":
    train_cifar100()