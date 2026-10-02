"""
Training script for IllumiCurveNet.

This script implements a training pipeline for our Low Light Image Enhancement model while supporting multiple loss configurations, early stopping,
and various training parameters that can be configured through command line arguments.

Key features:
- Dynamic loss configuration
- Early stopping mechanism
- Gradient clipping
- Model checkpointing
- Multiple loss functions for comprehensive image enhancement
"""

import os
import csv
import time
import argparse
import torch
import torch.optim
import model
import utils.losses as losses
import utils.dataloader as dataloader

def weights_init(m):
    """Initialize network weights using normal distribution.
    
    Args:
        m: Network module
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def compute_losses(IC_net, img_lowlight, weights, loss_fns):
    """Run the model on a batch and compute each weighted loss component.

    Returns (total_loss, component_dict) where component_dict holds the
    *unweighted* value of each active loss term, so per-component logs are
    comparable regardless of the weight configuration used.
    """
    L_color, L_spa, L_texture, L_exp, L_contrast, L_TV = loss_fns

    enhanced_image, A = IC_net(img_lowlight)

    components = {}
    if "L_color" in weights:
        components["L_color"] = torch.mean(L_color(enhanced_image))
    if "L_spa" in weights:
        components["L_spa"] = torch.mean(L_spa(enhanced_image, img_lowlight))
    if "L_TV" in weights:
        components["L_TV"] = torch.mean(L_TV(A))
    if "L_texture" in weights:
        components["L_texture"] = torch.mean(L_texture(img_lowlight, enhanced_image))
    if "L_exp" in weights:
        components["L_exp"] = torch.mean(L_exp(enhanced_image, img_lowlight))
    if "L_contrast" in weights:
        components["L_contrast"] = torch.mean(L_contrast(enhanced_image))

    total_loss = sum(weights[k] * v for k, v in components.items())
    return total_loss, components


def evaluate_loader(IC_net, loader, weights, loss_fns):
    """Compute average total loss and average per-component loss over a loader, no grad."""
    IC_net.eval()
    totals = {"total": 0.0}
    num_batches = 0
    with torch.no_grad():
        for img_lowlight in loader:
            img_lowlight = img_lowlight.cuda()
            loss, components = compute_losses(IC_net, img_lowlight, weights, loss_fns)
            if torch.isnan(loss).any():
                continue
            totals["total"] += loss.item()
            for k, v in components.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            num_batches += 1
    IC_net.train()
    if num_batches == 0:
        return {k: float('nan') for k in totals}
    return {k: v / num_batches for k, v in totals.items()}


def train(config):
    """Main training function.
    
    Args:
        config: ArgumentParser object containing training configurations
    """
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Initialize model to the GPU
    IC_net = model.illumi_curve_net().cuda()

    IC_net.apply(weights_init)
    if config.load_pretrain == True:
        IC_net.load_state_dict(torch.load(config.pretrain_snapshot))

    """
    Explanation of Weights:

    L_color (5.0): Set to a moderate weight to correct color discrepancies without dominating other losses.

    L_spa (1.5): Set higher weight to strongly enforce spatial consistency, crucial for maintaining sharpness and details.

    L_exp (10.0): Higher weight to adaptively enhance exposure based on input brightness, vital for varying low-light conditions.

    L_TV (200.0): Kept at a high weight to effectively suppress noise and artifacts, common in low-light images.

    L_contrast (5.0): Significant weight to improve visibility of details by enhancing local contrast.

    L_texture (3.0): Moderate weight to ensure texture details are preserved without introducing artifacts.
    """

    # Define the loss configuration. Weights are CLI-overridable (see --w_* below)
    # so different weight balances can be swept without editing this file.
    loss_configs = [
    {
        "name": "low_light_enhancement",
        "weights": {
            "L_color": config.w_color,
            "L_spa": config.w_spa,
            "L_exp": config.w_exp,
            "L_TV": config.w_tv,
            "L_contrast": config.w_contrast,
            "L_texture": config.w_texture,
        }
    }
]

    # Loop over each loss configuration
    for loss_config in loss_configs:
        print(f"Training with loss configuration: {loss_config['name']}")
        weights = loss_config["weights"]

        # Setup data loading
        train_dataset = dataloader.lowlight_loader(config.lowlight_images_path,
                                                    synthetic_degrade=config.synthetic_degrade)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                                   num_workers=config.num_workers, pin_memory=True)

        val_loader = None
        if config.val_lowlight_images_path:
            val_dataset = dataloader.lowlight_loader(config.val_lowlight_images_path,
                                                      synthetic_degrade=config.synthetic_degrade)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                                                      num_workers=config.num_workers, pin_memory=True)

        # Initialize all loss functions
        loss_fns = (
            losses.L_color(),
            losses.L_spa(),
            losses.L_texture(),
            losses.L_exp_masked(patch_size=16, mean_val=config.exp_mean_val, content_frac=config.exp_content_frac),
            losses.L_contrast(),
            losses.L_TV(),
        )

        # Setup optimizer
        optimizer = torch.optim.Adam(IC_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        IC_net.train()

        # Initialize early stopping parameters. Selection is based on validation
        # loss when a validation set is provided (more representative of
        # generalization), and falls back to training loss otherwise.
        best_loss = float('inf')
        patience = config.early_stopping_patience
        patience_counter = 0
        min_delta = 1e-4

        os.makedirs(config.log_dir, exist_ok=True)
        log_path = os.path.join(config.log_dir, f"training_log_{config.run_name}.csv")
        component_names = list(weights.keys())
        log_fieldnames = ["epoch", "epoch_time_sec", "train_loss_total"] + \
            [f"train_{k}" for k in component_names]
        if val_loader is not None:
            log_fieldnames += ["val_loss_total"] + [f"val_{k}" for k in component_names]
        with open(log_path, 'w', newline='') as log_file:
            csv.DictWriter(log_file, fieldnames=log_fieldnames).writeheader()
        print(f"Logging per-epoch metrics to {log_path}")

        # Main training loop
        for epoch in range(config.num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_components = {k: 0.0 for k in component_names}
            num_batches = 0

            for iteration, img_lowlight in enumerate(train_loader):
                img_lowlight = img_lowlight.cuda()

                loss, components = compute_losses(IC_net, img_lowlight, weights, loss_fns)

                # Handle NaN losses
                if torch.isnan(loss).any():
                    print("NaN detected in loss. Skipping iteration.")
                    continue

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(IC_net.parameters(), config.grad_clip_norm)
                optimizer.step()

                epoch_loss += loss.item()
                for k, v in components.items():
                    epoch_components[k] += v.item()
                num_batches += 1

                # Progress display
                if (iteration + 1) % config.display_iter == 0:
                    print(f"Epoch [{epoch+1}/{config.num_epochs}], Iteration [{iteration+1}], Loss: {loss.item()}")

                # Save model checkpoint
                if (iteration + 1) % config.checkpoint_iter == 0:
                    checkpoint_name = f"model-{loss_config['name']}-epoch{epoch}-iteration{iteration}.pth"
                    torch.save(IC_net.state_dict(), os.path.join(config.checkpoints_folder, checkpoint_name))

            # Calculate average epoch loss and per-component averages
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_components = {k: v / num_batches for k, v in epoch_components.items()}

            # Compute validation metrics for this epoch, if configured
            val_metrics = None
            if val_loader is not None:
                val_metrics = evaluate_loader(IC_net, val_loader, weights, loss_fns)
                print(f"Epoch [{epoch+1}/{config.num_epochs}] train_loss={avg_epoch_loss:.4f} val_loss={val_metrics['total']:.4f}")

            # Log this epoch's metrics to CSV
            log_row = {
                "epoch": epoch + 1,
                "epoch_time_sec": time.time() - epoch_start,
                "train_loss_total": avg_epoch_loss,
            }
            log_row.update({f"train_{k}": v for k, v in avg_epoch_components.items()})
            if val_metrics is not None:
                log_row["val_loss_total"] = val_metrics["total"]
                log_row.update({f"val_{k}": val_metrics[k] for k in component_names})
            with open(log_path, 'a', newline='') as log_file:
                csv.DictWriter(log_file, fieldnames=log_fieldnames).writerow(log_row)

            # Early stopping logic (validation loss when available, else training loss)
            selection_loss = val_metrics["total"] if val_metrics is not None else avg_epoch_loss
            if selection_loss < best_loss - min_delta:
                best_loss = selection_loss
                patience_counter = 0
                # Save best model
                best_model_name = f"model-best-{config.run_name}.pth"
                torch.save(IC_net.state_dict(), os.path.join(config.snapshots_folder, best_model_name))
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        print(f"Finished training with loss configuration: {loss_config['name']}")


if __name__ == "__main__":
    # Setup command line argument parser
    parser = argparse.ArgumentParser(description='IllumiCurveNet Training Script')

    # Training configuration parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/", help='Path to low-light training images')
    parser.add_argument('--val_lowlight_images_path', type=str, default="", help='Path to validation images; if empty, no validation is run')
    parser.add_argument('--log_dir', type=str, default="logs/", help='Directory to write per-epoch metrics CSV')
    parser.add_argument('--run_name', type=str, default="v1", help='Identifier used in the best-model filename and metrics log filename')
    parser.add_argument('--synthetic_degrade', type=lambda s: s.lower() != 'false', default=True,
                         help='Apply random synthetic low-light degradation (gamma+exposure+noise) to training/val images')
    parser.add_argument('--exp_mean_val', type=float, default=0.6, help='Target patch brightness for the masked exposure loss')
    parser.add_argument('--exp_content_frac', type=float, default=0.15,
                         help='Fraction of an image\'s brightest patch above which a patch counts as "content" for the exposure loss')
    parser.add_argument('--w_color', type=float, default=5.0, help='Weight for L_color')
    parser.add_argument('--w_spa', type=float, default=1.5, help='Weight for L_spa')
    parser.add_argument('--w_exp', type=float, default=10.0, help='Weight for L_exp')
    parser.add_argument('--w_tv', type=float, default=200.0, help='Weight for L_TV')
    parser.add_argument('--w_contrast', type=float, default=5.0, help='Weight for L_contrast')
    parser.add_argument('--w_texture', type=float, default=3.0, help='Weight for L_texture')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for optimizer')
    parser.add_argument('--grad_clip_norm', type=float, default=1, help='Gradient clipping norm')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=4, help='Validation batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--display_iter', type=int, default=5, help='Display loss every N iterations')
    parser.add_argument('--checkpoint_iter', type=int, default=5, help='Save model every N iterations')
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/", help='Directory to save best model')
    parser.add_argument('--checkpoints_folder', type=str, default="checkpoints/", help='Directory to save model checkpoints')
    parser.add_argument('--load_pretrain', type=bool, default= False, help='Whether to load pretrained model')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Number of epochs to wait before early stopping')
    parser.add_argument('--pretrain_snapshot', type=str, default= "snapshots/model-best.pth", help='Pretrained model snapshot')

    config = parser.parse_args()

    # Create snapshots and checkpoints directories if they don't exist
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.checkpoints_folder):
        os.mkdir(config.checkpoints_folder)

    train(config)