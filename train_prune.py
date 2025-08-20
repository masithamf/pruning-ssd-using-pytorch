import argparse
import os
import logging
import sys
import itertools
import re
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.optim import Adam

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config, mobilenetv1_ssd_config, squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform


# Determine output folder for this run
base_save_dir = "prunedmodel"
# os.makedirs(base_save_dir, exist_ok=True)
# existing_runs = sorted([d for d in os.listdir(base_save_dir) if os.path.isdir(os.path.join(base_save_dir, d)) and d.startswith("pruning")])
# run_id = len(existing_runs) + 1
save_dir = os.path.join(base_save_dir, f"pruning6")
os.makedirs(save_dir, exist_ok=True)

def visualize_layer_sparsity(model, sparsity_before, prune_amount=0.2, save_path=None):
    layer_names = []
    total_weights = []
    nonzero_weights = []

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            total = module.weight.nelement()
            nonzero = torch.count_nonzero(module.weight).item()
            total_weights.append(total)
            nonzero_weights.append(nonzero)
            layer_names.append(name)

    x = np.arange(len(layer_names))
    plt.figure(figsize=(14, 6))
    plt.plot(x, total_weights, marker='o', label='Total Weights Before Pruning')
    plt.plot(x, nonzero_weights, marker='x', label='Nonzero Weights After Pruning')
    plt.xticks(x, layer_names, rotation=90, fontsize=7)
    plt.ylabel("# Weights")
    plt.title(f"Layer Weight Count Comparison (Prune {prune_amount * 100:.0f}%)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Saved sparsity visualization to {save_path}")
    else:
        plt.show()
        plt.savefig(save_path)
        logging.info(f"Saved sparsity visualization to {save_path}")

def apply_global_pruning(model, amount=0.2):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    logging.info("Pruned the following parameters:")
    for module, param in parameters_to_prune:
        total_weights = module.weight.nelement()
        zero_weights = torch.sum(module.weight == 0).item()
        sparsity = 100.0 * zero_weights / total_weights
        logging.info(f"{module.__class__.__name__} | Sparsity: {sparsity:.2f}% ({zero_weights}/{total_weights})")

    logging.info("Visualizing sparsity after pruning...")
    sparsity_before = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            total = module.weight.nelement()
            zeros = torch.sum(module.weight == 0).item()
            sparsity_before.append(0.0)  # assume no zeros before actual pruning
    visualize_layer_sparsity(model, sparsity_before, prune_amount=amount, save_path=os.path.join(save_dir, "sparsity_after_pruning.png"))

    return parameters_to_prune

def remove_pruning_masks(model):
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss, running_regression_loss, running_classification_loss = 0.0, 0.0, 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images, boxes, labels = images.to(device), boxes.to(device), labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

        if i and i % debug_steps == 0:
            logging.info(f"Epoch: {epoch}, Step: {i}, Average Loss: {running_loss / debug_steps:.4f}, "
                         f"Reggression Loss: {running_regression_loss / debug_steps:.4f}, Classification Loss: {running_classification_loss / debug_steps:.4f}")
            running_loss, running_regression_loss, running_classification_loss = 0.0, 0.0, 0.0

def test(loader, net, criterion, device):
    net.eval()
    running_loss, running_regression_loss, running_classification_loss = 0.0, 0.0, 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images, boxes, labels = images.to(device), boxes.to(device), labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            running_loss += (regression_loss + classification_loss).item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSD Training with Pruning')
    parser.add_argument("--dataset_type", default="voc")
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--validation_dataset')
    parser.add_argument('--balance_data', action='store_true')
    parser.add_argument('--net', default="vgg16-ssd")
    parser.add_argument('--freeze_base_net', action='store_true')
    parser.add_argument('--freeze_net', action='store_true')
    parser.add_argument('--mb2_width_mult', default=1.0, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--base_net_lr', default=None, type=float)
    parser.add_argument('--extra_layers_lr', default=None, type=float)
    parser.add_argument('--base_net')
    parser.add_argument('--pretrained_ssd')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--scheduler', default="multi-step")
    parser.add_argument('--milestones', default="80,100")
    parser.add_argument('--t_max', default=120, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--validation_epochs', default=5, type=int)
    parser.add_argument('--debug_steps', default=100, type=int)
    parser.add_argument('--use_cuda', default=True, type=str2bool)
    parser.add_argument('--checkpoint_folder', default='models/')
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--prune_amount', default=0.2, type=float)

    args = parser.parse_args()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    timer = Timer()
    logging.info(args)

    config = vgg_ssd_config
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3-large-ssd-lite':
        create_net = create_mobilenetv3_large_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3-small-ssd-lite':
        create_net = create_mobilenetv3_small_ssd_lite
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("Unsupported net type.")
        sys.exit(1)

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)

    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform, target_transform=target_transform)
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        elif args.dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path, transform=train_transform,
                                         target_transform=target_transform, dataset_type="train",
                                         balance_data=args.balance_data)
            label_file = os.path.join(args.checkpoint_folder, "open-images-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        datasets.append(dataset)

    train_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True)

    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    else:
        val_dataset = OpenImagesDataset(args.validation_dataset, transform=test_transform,
                                        target_transform=target_transform, dataset_type="test")
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False)

    net = create_net(num_classes)

    if args.prune:
        logging.info(f"Applying global pruning with amount = {args.prune_amount}")
        apply_global_pruning(net, amount=args.prune_amount)

    base_net_lr = args.base_net_lr if args.base_net_lr else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr else args.lr

    if args.freeze_base_net:
        freeze_net_layers(net.base_net)
        params = [
            {'params': itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters()), 'lr': extra_layers_lr},
            {'params': itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters()), 'lr': extra_layers_lr},
            {'params': itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())}
        ]

    if args.resume:
        logging.info(f"Resuming from {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        net.init_from_pretrained_ssd(args.pretrained_ssd)

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    last_epoch = -1
    if args.resume:
        match = re.search(r"Epoch-(\d+)", args.resume)
        if match:
            last_epoch = int(match.group(1))

    for epoch in range(last_epoch + 1, args.num_epochs):
        train(train_loader, net, criterion, optimizer, DEVICE, args.debug_steps, epoch)

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_reg_loss, val_cls_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(f"Epoch: {epoch}, Val Loss: {val_loss:.4f}, Reggression Loss: {val_reg_loss:.4f}, Classification Loss: {val_cls_loss:.4f}")
            remove_pruning_masks(net)
            model_path = os.path.join(save_dir, f"Prune-{args.net}-Epoch-{epoch}-Loss-{val_loss:.4f}.pth")
            torch.save(net.state_dict(), model_path)
            logging.info(f"Saved pruned model state_dict to {model_path}")
