"""
Loss functions for image enhancement tasks.

This module contains various loss functions used for training image enhancement models.
The loss functions are implemented as PyTorch modules and include:
- Color loss (L_color)
- Spatial loss (L_spa)
- Exposure loss (L_exp)
- Total Variation loss (L_TV)
- Saturation loss (Sa_Loss)
- Perceptual loss (perception_loss)
- Texture loss (L_texture)
- Dynamic exposure loss (L_exp_dynamic)
- Contrast loss (L_contrast)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16

class L_color(nn.Module):
    """Color loss to maintain color consistency."""
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b,c,h,w = x.shape

        # Calculate mean RGB values
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        # Calculate color differences
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

        return k

class L_spa(nn.Module):
    """Spatial loss to preserve spatial consistency."""
    def __init__(self):
        super(L_spa, self).__init__()
        # Define directional kernels for edge detection
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b,c,h,w = org.shape

        # Calculate mean values
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        # Pool features
        org_pool = self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        # Calculate weight differences
        weight_diff = torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)

        # Apply directional convolutions
        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        # Calculate directional differences
        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)

        return E

class L_exp(nn.Module):
    """Exposure loss to control the overall brightness."""
    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val]).cuda(),2))
        return d

class L_exp_masked(nn.Module):
    """Exposure loss restricted to patches that actually contain scene content.

    Plain L_exp pulls every patch's mean brightness toward mean_val, which
    assumes the whole frame should end up well-lit. For a mostly-black-sky
    scene (e.g. a moon frame) that fights the background: most patches are
    sky and have nothing to "expose". Here each patch is compared against the
    same input image's own brightest patch, and only patches at least
    content_frac of that peak (i.e. the illuminated subject, not empty sky)
    are pulled toward mean_val.
    """
    def __init__(self, patch_size=16, mean_val=0.6, content_frac=0.15):
        super(L_exp_masked, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
        self.content_frac = content_frac

    def forward(self, enhanced, img_lowlight):
        enhanced_gray = torch.mean(enhanced, 1, keepdim=True)
        input_gray = torch.mean(img_lowlight, 1, keepdim=True)

        enhanced_patch = self.pool(enhanced_gray)
        input_patch = self.pool(input_gray)

        b = input_patch.shape[0]
        max_patch = input_patch.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        mask = (input_patch >= self.content_frac * max_patch).float()

        diff_sq = torch.pow(enhanced_patch - self.mean_val, 2) * mask
        denom = mask.view(b, -1).sum(dim=1).clamp(min=1.0)
        per_image = diff_sq.view(b, -1).sum(dim=1) / denom
        return per_image.mean()

class L_TV(nn.Module):
    """Total Variation loss to ensure smoothness."""
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        # Calculate horizontal and vertical differences
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class Sa_Loss(nn.Module):
    """Saturation loss to maintain color saturation."""
    def __init__(self):
        super(Sa_Loss, self).__init__()

    def forward(self, x):
        b,c,h,w = x.shape
        # Split channels and calculate mean
        r,g,b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg,mb = torch.split(mean_rgb, 1, dim=1)
        # Calculate channel differences
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k = torch.pow(torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        k = torch.mean(k)
        return k

class perception_loss(nn.Module):
    """Perceptual loss using VGG16 features."""
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        # Create feature extractors for different layers
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Extract features from different layers
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = h
        return out

class L_texture(nn.Module):
    """Texture loss to preserve image details."""
    def __init__(self):
        super(L_texture, self).__init__()

    def forward(self, img_lowlight, img_enhanced):
        # Convert images to grayscale
        img_lowlight_gray = torch.mean(img_lowlight, dim=1, keepdim=True)
        img_enhanced_gray = torch.mean(img_enhanced, dim=1, keepdim=True)

        # Define Sobel filters
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)

        # Calculate gradients
        grad_low_x = F.conv2d(img_lowlight_gray, sobel_x, padding=1)
        grad_low_y = F.conv2d(img_lowlight_gray, sobel_y, padding=1)
        grad_enhanced_x = F.conv2d(img_enhanced_gray, sobel_x, padding=1)
        grad_enhanced_y = F.conv2d(img_enhanced_gray, sobel_y, padding=1)

        # Calculate gradient magnitudes
        grad_low = torch.sqrt(grad_low_x ** 2 + grad_low_y ** 2 + 1e-8)
        grad_enhanced = torch.sqrt(grad_enhanced_x ** 2 + grad_enhanced_y ** 2 + 1e-8)

        loss = torch.mean((grad_low - grad_enhanced) ** 2)
        return loss

class L_exp_dynamic(nn.Module):
    """Dynamic exposure loss based on input image brightness."""
    def __init__(self, patch_size=16):
        super(L_exp_dynamic, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, img_enhanced, img_lowlight):
        # Calculate dynamic target based on input image
        dynamic_target = torch.mean(img_lowlight)
        mean_patch = self.pool(img_enhanced)
        loss = torch.mean((mean_patch - dynamic_target) ** 2)
        return loss

class L_contrast(nn.Module):
    """Contrast loss to maintain proper local contrast."""
    def __init__(self, patch_size=16, ideal_contrast=0.5):
        super(L_contrast, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.ideal_contrast = ideal_contrast

    def forward(self, img_enhanced):
        # Calculate local contrast
        mean_patch = self.pool(img_enhanced)
        mean_patch_upsampled = F.interpolate(mean_patch, size=img_enhanced.shape[2:], mode='nearest')
        contrast = self.pool((img_enhanced - mean_patch_upsampled) ** 2)
        contrast = torch.sqrt(contrast + 1e-8)
        loss = torch.mean((contrast - self.ideal_contrast) ** 2)
        return loss
