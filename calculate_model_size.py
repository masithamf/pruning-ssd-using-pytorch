import torch
import os
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.config import mobilenetv1_ssd_config

def load_pruned_model(pth_path, num_classes=21, width_mult=1.0, device='cpu'):
    model = create_mobilenetv2_ssd_lite(num_classes=num_classes, width_mult=width_mult)
    model.load(pth_path)
    model.to(device)
    model.eval()
    return model

def report_sparsity_per_layer(model):
    report = []
    total_params = 0
    total_zeros = 0

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            weight = module.weight.data
            num_params = weight.numel()
            num_zeros = torch.sum(weight == 0).item()
            sparsity = 100.0 * num_zeros / num_params

            report.append((name, num_params, num_zeros, sparsity))
            total_params += num_params
            total_zeros += num_zeros

    print(f"{'Layer':40} {'#Params':>10} {'#Zeros':>10} {'Sparsity (%)':>15}")
    for name, total, zeros, sp in report:
        print(f"{name:40} {total:10d} {zeros:10d} {sp:15.2f}")
    print(f"\nTOTAL PARAMS: {total_params:,}, TOTAL ZEROS: {total_zeros:,}, GLOBAL SPARSITY: {100.0 * total_zeros / total_params:.2f}%")

def check_model_file_size(path):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"File size of model: {size_mb:.2f} MB")

if __name__ == '__main__':
    model_path = "prunedmodel/pruning1/amount10-200-loss-3.5432.pth"  # Ganti sesuai file kamu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 4  
    width_mult = 1.0 

    model = load_pruned_model(model_path, num_classes=num_classes, width_mult=width_mult, device=device)
    report_sparsity_per_layer(model)
    check_model_file_size(model_path)