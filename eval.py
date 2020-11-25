import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.metrics import SegmentationMetric

num_classes = 20
metric = SegmentationMetric(num_classes)


from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        val_fwiou_sum = 0.0
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            tot += F.cross_entropy(mask_pred, true_masks.squeeze(1)).item()

            for i in range(mask_pred.shape[0]):
                pre = mask_pred[i, :, :, :].argmax(axis=0).cpu()
                label = true_masks.squeeze(1)[i, :, :].cpu()
                metric.addBatch(pre, label)
                FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
                val_fwiou_sum += FWIoU
            n = true_masks.shape[0]
            pbar.update()

    net.train()
    return tot / n_val,  val_fwiou_sum / n