'''
Notes:
- Lightning should automatically handle gradscaler, autocast etc.
- InfoNCE, aandoning posweight and rebalancing of the loss. This should be ok.

'''

dino_model = "facebook/dinov3-vitl16-pretrain-lvd1689m"
class PostDinoConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        # Standard non-linear projection head for contrastive learning
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (B, input_dim)
        x = self.mlp(x)
        # Match original DINO embedding normalization space
        return F.normalize(x, p=2, dim=1)

# Get DINO output embedding dimension
dino_output_dim = dino_model.config.hidden_size
# Initialize the post-DINO convolutional layer
post_dino_conv_layer = PostDinoConvLayer(input_dim=dino_output_dim).to(device)


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Helper function required by torch.utils.checkpoint
def run_dino_checkpoint(chunk, backbone):
    outputs = backbone(chunk)
    # If using Hugging Face ViT/DINO, outputs.last_hidden_state contains all tokens.
    # We extract the [CLS] token (index 0) to match the original ViTEncoder behavior.
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, 0, :]
    return outputs[:, 0, :]
# This function agrees with repo. Originally used pooler method
'''
def run_dino_checkpoint(chunk, backbone):
    # Use pooler output to preserve original notebook's space representation[cite: 1]
    outputs = backbone(pixel_values=chunk).pooler_output
    return F.normalize(outputs.float(), p=2, dim=1).to(torch.float32)
'''


class InOutEncoder(nn.Module): # 1. Must inherit from nn.Module
    def __init__(self, dino_model, chunk_size=4, num_channels=7, projection_size=512): # Can manually define model name, should be string title of dino model.
        super().__init__()
        self.chunk_size = chunk_size

        #ADDED:
        # Pull processor configs dynamically to extract original mean/std
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(dino_model)
        # Register normalization statistics as buffers so they move with .to(device) automatically
        self.register_buffer("mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 3, 1, 1))

        # --- PRE-DINO: INPUT CNN ---
        # Reduces your custom 7 channels down to the 3 channels DINO expects
        self.bn = nn.BatchNorm2d(num_channels)
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        

        # BACKBONE & POST-HEAD
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(dino_model)
        self.backbone.config.use_cache = False
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.post_dino_layer = PostDinoConvLayer(input_dim=self.backbone.config.hidden_size, output_dim=projection_size)
        
        '''
        # --- CORE: BACKBONE ---
        self.backbone = dino_model
        # Freeze DINO parameters completely
        for param in self.backbone.parameters():
            param.requires_grad = False

        # --- POST-DINO: PROJECTION HEAD ---
        # Get hidden size dynamically from DINO's configuration
        dino_output_dim = self.backbone.config.hidden_size
        self.post_dino_layer = PostDinoConvLayer(
            input_dim=dino_output_dim, 
            output_dim=projection_size
        )
        '''
    def forward(self, x):
        # x shape: (B, 7, H, W)
        
        # 1. Run Pre-DINO layer to transform 7 channels to 3 channels
        x = self.bn(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)  # ADDED
        # pixel_values = self.conv3(x)  # Now shape is (B, 3, H, W)

        # 2. ACCURACY OPTIMIZATION: Differentiable Preprocessing
        x = torch.sigmoid(x)
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        pixel_values = (x - self.mean) / self.std

        # 2. Chunking & Gradient Checkpointing over the Batch dimension
        dino_embeddings_list = []
        num_images = pixel_values.size(0)
        for start in range(0, num_images, self.chunk_size):
            chunk = pixel_values[start : start + self.chunk_size]
            
            # Use gradient checkpointing to save memory during training
            chunk_emb = checkpoint(
                run_dino_checkpoint,
                chunk,
                self.backbone,
                use_reentrant=False
            )
            dino_embeddings_list.append(chunk_emb)
            
        # Combine chunks back together into shape: (B, dino_output_dim)
        dino_embeddings_flat = torch.cat(dino_embeddings_list, dim=0)
        
        # 3. Project to final contrastive space & return normalized vector
        output_embeddings = self.post_dino_layer(dino_embeddings_flat)
        
        return output_embeddings