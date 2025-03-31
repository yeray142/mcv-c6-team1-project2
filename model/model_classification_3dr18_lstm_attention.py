"""
File containing the main model.

- This model uses a 3D ResNet architecture for video classification. (3D-backbone)
- It includes augmentations and normalization specific to video data.
- Includes LSTM and attention layers for temporal processing. (LSTM + attention)
"""

#Standard imports
import torch
from torch import nn
import torchvision
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class Model(BaseRGBModel):
    class Impl(nn.Module):
        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            # Replace 2D CNN with 3D ResNet (new code)
            if self._feature_arch.startswith('3dresnet') or self._feature_arch.startswith('r3d_18'):
                print('Using 3D ResNet-18')
                self._features = torchvision.models.video.r3d_18(pretrained=True)
            elif self._feature_arch.startswith('mc3_18'):
                print('Using MC3-18')
                self._features = torchvision.models.video.mc3_18(pretrained=True)
            elif self._feature_arch.startswith('r2plus1d_18'):
                print('Using R2Plus1D-18')
                self._features = torchvision.models.video.r2plus1d_18(pretrained=True)
            self._d = 512
            
            # Important: Replace the FC layer with Identity to get features
            # self._features.fc = nn.Identity()
            
            # Add LSTM for temporal processing (similar to the second model)
            self._lstm = nn.LSTM(input_size=self._d, hidden_size=self._d, 
                                batch_first=True, num_layers=1, bidirectional=True)
            lstm_out_dim = self._d * 2  # because of bidirectionality
            
            # Add attention layer for better temporal modeling
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=lstm_out_dim,  # Matches LSTM's output dimension
                num_heads=4,            # 4 attention heads
                batch_first=True
            )
            
            # Final classification layers
            self._fc = FCLayers(lstm_out_dim, args.num_classes)

            # Update normalization for video models (critical change)
            self.standarization = T.Compose([
                T.Normalize(mean = (0.43216, 0.394666, 0.37645), 
                            std = (0.22803, 0.22145, 0.216989)) # Kinetics-400 stats
            ])

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape # B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch
            x = self.standarize(x) # Standarization Kinetics-400 stats
            
            # Reformat input for 3D CNN: (B,T,C,H,W) -> (B,C,T,H,W)
            x = x.permute(0, 2, 1, 3, 4)
            
            # Get 3D CNN features without pooling
            # We need to modify the forward call to extract features without pooling
            x = self._features.stem(x)
            x = self._features.layer1(x)
            x = self._features.layer2(x)
            x = self._features.layer3(x)
            x = self._features.layer4(x)
            
            # Now x has shape (B, C, T', H', W')
            # Pooling spatial dimensions only, keeping temporal dimension
            x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))  # (B, C, T', 1, 1)
            x = x.squeeze(-1).squeeze(-1)  # (B, C, T')
            
            # Rearrange to (B, T', C) for LSTM
            x = x.permute(0, 2, 1)  # (B, T', C)
            
            # Pass through LSTM
            x, _ = self._lstm(x)  # output shape: (B, T', 2*_d)
            
            # Apply attention
            attn_out, _ = self.attention_layer(x, x, x)
            
            # Global temporal pooling
            x = torch.mean(attn_out, dim=1)  # (B, 2*_d)

            # Final classification
            x = self._fc(x)  # (B, num_classes)
            return x
        
        def normalize(self, x):
            return x / 255.
        
        def augment(self, x):
            # Original augmentations but applied consistently across frames
            # x shape: (B, T, C, H, W)
            
            # Generate random parameters ONCE per clip
            flip_prob = torch.rand(x.size(0)) < 0.5  # (B,) boolean tensor
            jitter_params = torch.rand(x.size(0), 4) # (B, 4) for brightness, contrast, saturation, hue
            
            # Apply same augmentation to all frames in a clip
            for b in range(x.size(0)):
                # Horizontal flip (same for all frames)
                if flip_prob[b]:
                    x[b] = T.functional.hflip(x[b])
                
                # Color jitter (same parameters for all frames)
                x[b] = T.functional.adjust_brightness(x[b], jitter_params[b,0] * 0.2 + 0.9)  # 0.9-1.1
                x[b] = T.functional.adjust_contrast(x[b], jitter_params[b,1] * 0.2 + 0.9)
                x[b] = T.functional.adjust_saturation(x[b], jitter_params[b,2] * 0.2 + 0.9)
                x[b] = T.functional.adjust_hue(x[b], jitter_params[b,3] * 0.1 - 0.05)  # -0.05-0.05
                
                # Gaussian blur (same for all frames)
                if torch.rand(1) < 0.25:
                    x[b] = T.functional.gaussian_blur(x[b], kernel_size=5)
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).float()
                
                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    
                    # Apply loss function
                    assert self._args.loss in ['bce', 'focal'], "Loss function must be 'bce' or 'focal'"
                    if self._args.loss == 'focal':
                        assert self._args.alpha is not None, "Alpha must be set for focal loss"
                        assert self._args.gamma is not None, "Gamma must be set for focal loss"
                        loss = sigmoid_focal_loss(pred, label, alpha=self._args.alpha, gamma=self._args.gamma,reduction='mean')
                    elif self._args.loss == 'bce':
                        loss = F.binary_cross_entropy_with_logits(pred, label)
                    
                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)
                epoch_loss += loss.detach().item()
        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # apply sigmoid
            pred = torch.sigmoid(pred)
            
            return pred.cpu().numpy()
