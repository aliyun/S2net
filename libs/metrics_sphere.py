import os
import torch, typing
import functools
from .logger import *
from .metrics import AverageMeter

# we use official evaluation implementation of Pano3D dataset(https://github.com/VCL3D/Pano3D)

from mapped_convolution.util import (
    generate_icosphere,
    sphere_to_image_resample_map,
)
from mapped_convolution.nn import (
    Resample,
    Unresample
)


def _dim_list(
    tensor:         torch.Tensor,
    start_index:    int=1,
) -> typing.List[int]:
    return list(range(start_index, len(tensor.shape)))


class Delta(torch.nn.Module):
    def __init__(self,
        threshold: float=1.25
    ):
        super(Delta, self).__init__()
        self.threshold = threshold

    def compute(self,
        gt:             torch.Tensor,
        pred:           torch.Tensor,
        weights:        torch.Tensor=None,
    ) -> torch.Tensor: #NOTE: no mean
        errors =  (torch.max((gt / pred), (pred / gt)) < self.threshold).float()
        if weights is None:
            return torch.mean(errors)
        else:
            return torch.mean(
                torch.sum(errors * weights, dim=_dim_list(gt))
                / torch.sum(weights, dim=_dim_list(gt))
            )

Delta_11 = functools.partial(Delta, threshold=1.1)
Delta1 = functools.partial(Delta, threshold=1.25)
Delta2 = functools.partial(Delta, threshold=1.25 ** 2)
Delta3 = functools.partial(Delta, threshold=1.25 ** 3)


class AbsRel(torch.nn.Module):
    def __init__(self):
        super(AbsRel, self).__init__()

    def compute(self,
        gt:         torch.Tensor,
        pred:       torch.Tensor,
        weights:    torch.Tensor=None,
        mask:       torch.Tensor=None,
    ) -> torch.Tensor:
        absrel = torch.abs((gt - pred) / gt )
        if weights is not None:
            absrel = absrel * weights
        if mask is not None:
            absrel = absrel[mask]
        if weights is None:
            return torch.mean(torch.mean(absrel, dim=_dim_list(gt)))
        else:
            return torch.mean(
                torch.sum(absrel, dim=_dim_list(gt))
                / torch.sum(weights, dim=_dim_list(gt))
            )


class RMSE(torch.nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def compute(self,
        gt:         torch.Tensor,
        pred:       torch.Tensor,
        weights:    torch.Tensor=None,
        mask:       torch.Tensor=None,
    ) -> torch.Tensor:
        diff_sq = (gt - pred) ** 2
        if weights is not None:
            diff_sq = diff_sq * weights
        if mask is not None:
            diff_sq = diff_sq[mask]
        if weights is None:
            return torch.mean(torch.sqrt(torch.mean(diff_sq, dim=_dim_list(gt))))
        else:
            diff_sq_sum = torch.sum(diff_sq, dim=_dim_list(gt))
            diff_w_sum = torch.sum(weights, dim=_dim_list(gt)) # + 1e-18
            return torch.mean(torch.sqrt(diff_sq_sum / diff_w_sum))


class RMSLE(torch.nn.Module):
    def __init__(self,
        base: str='ten' # 'natural' or 'ten'
    ):
        super(RMSLE, self).__init__()
        self.log = torch.log if base == 'natural' else torch.log10

    def compute(self,
        gt:         torch.Tensor,
        pred:       torch.Tensor,
        weights:    torch.Tensor=None,
        mask:       torch.Tensor=None,
    ) -> torch.Tensor:
        pred_fix = torch.where(pred == 0.0,
            pred + 1e-24,
            pred
        )
        log_diff_sq = (self.log(gt) - self.log(pred_fix)) ** 2
        if weights is not None:
            log_diff_sq = log_diff_sq * weights
        if mask is not None:
            log_diff_sq = log_diff_sq[mask]
        if weights is None:
            return torch.mean(torch.sqrt(torch.mean(log_diff_sq, dim=_dim_list(gt))))
        else:
            log_diff_sq_sum = torch.sum(log_diff_sq, dim=_dim_list(gt))
            log_diff_w_sum = torch.sum(weights, dim=_dim_list(gt)) # + 1e-18
            return torch.mean(torch.sqrt(log_diff_sq_sum / log_diff_w_sum))


class SqRel(torch.nn.Module):
    def __init__(self):
        super(SqRel, self).__init__()

    def compute(self,
        gt:         torch.Tensor,
        pred:       torch.Tensor,
        weights:    torch.Tensor=None,
        mask:       torch.Tensor=None,
    ) -> torch.Tensor:
        sqrel = ((gt - pred) ** 2) / gt
        if weights is not None:
            sqrel = sqrel * weights
        if mask is not None:
            sqrel = sqrel[mask]
        if weights is None:
            return torch.mean(torch.mean(sqrel, dim=_dim_list(gt)))
        else:
            return torch.mean(
                torch.sum(sqrel, dim=_dim_list(gt))
                / torch.sum(weights, dim=_dim_list(gt))
            )


class Sample(Resample):
    def __init__(self,
        order:          int=5,
        width:          int=512,
        interpolation:  str='bispherical', # one of ['bispherical', 'nearest']
        layout:         str='mesh', # one of ['image' or 'mesh']
        persistent:     bool=True,
        epsilon:        float=1.0e-12,
    ):
        super(Sample, self).__init__(interpolation)
        self.layout = layout
        self.num_vertices = generate_icosphere(order).num_vertices()
        sample_map, interp_map = sphere_to_image_resample_map(
            order, (width // 2, width), interpolation == 'nearest')

        sum_weights = torch.zeros(self.num_vertices)
        sum_weights.index_add_(
            0,
            sample_map[..., 0].long().view(-1),
            interp_map.view(-1)
        )
        self.register_buffer("sample_map", sample_map, persistent=persistent)
        self.register_buffer("interp_map", interp_map, persistent=persistent)
        self.register_buffer("sum_weights", sum_weights + epsilon, persistent=persistent)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        vertices = super(Sample, self).forward(
            image, self.sample_map, (1, self.num_vertices), self.interp_map
        )
        vertices = vertices / self.sum_weights
        if self.layout == 'mesh':
            vertices = vertices.squeeze(-2)
            vertices = vertices.transpose(-1, -2)
        # if not torch.isfinite(vertices).all():
        #     verts_finite = torch.isfinite(vertices).sum()
            # image_finite = torch.isfinite(image).sum()
            # print(f'\t V = {verts_finite} ({verts_finite / vertices.nelement()} %) \t I = {image_finite} ({image_finite / image.nelement()} %)')
        # print(image.requires_grad)
        return vertices


SampleLinear = functools.partial(Sample, interpolation='bispherical', layout='image')
SampleMeshLinear = functools.partial(Sample, interpolation='bispherical', layout='mesh')
SampleNearest = functools.partial(Sample, interpolation='nearest', layout='mesh')


@torch.no_grad()
def compute_depth_spherical_metrics(gt, pred, mask=None, median_align=False, min_depth=0.1, max_depth=10.0, weight=None):

    """ Computing depth estimation considering spherical weights.

        Computing depth metrics weighting each pixel by its spherical position.
        If weight is None, all pixels are treated equally

        Args:
            gt: ground truth depth map.
            pred: predicted depth map.
            mask: mask valid area in depth map
            median_align: do scale alignment by their median depth between gt and pred.
            min_depth: valid minimum depth range.
            max_depth: valid maximum depth range.
            weight: weighting each pixel by its spherical position.
        Returns:
            a11, a1, a2, a3: the accuracy metrics, corresponding to delta_1.1, delta_1.25, delta_1.25^2, delta_1.25^3
    """

    if mask is None:
        mask = (gt > min_depth) & (gt <= max_depth)

    gt = gt[mask]
    pred = pred[mask]

    if weight is not None:
        weight = weight[mask]
    else:
        weight = torch.ones_like(gt).to(gt.device)

    # truncated depth
    gt[gt < min_depth] = min_depth
    gt[gt > max_depth] = max_depth
    pred[pred < min_depth] = min_depth
    pred[pred > max_depth] = max_depth

    if median_align:
        pred *= torch.median(gt) / torch.median(pred)
    delta11, delta1, delta2, delta3 = Delta_11(), Delta1(), Delta2(), Delta3()
    a1 = delta1.compute(gt=gt, pred=pred, weights=weight)
    a2 = delta2.compute(gt=gt, pred=pred, weights=weight)
    a3 = delta3.compute(gt=gt, pred=pred, weights=weight)
    a11 = delta11.compute(gt=gt, pred=pred, weights=weight)

    rmse_solver = RMSE()
    rmse = rmse_solver.compute(gt=gt, pred=pred, weights=weight)
    # rmse = (gt - pred) ** 2
    # rmse = torch.sqrt(torch.sum(weight * rmse, dim=_dim_list(gt)).mean())

    rmse_log_solver = RMSLE()
    rmse_log = rmse_log_solver.compute(gt=gt, pred=pred, weights=weight)
    # rmse_log = (torch.log10(gt) - torch.log10(pred)) ** 2
    # rmse_log = torch.sqrt((weight * rmse_log).mean())

    abs = torch.sum(weight * torch.abs(gt - pred)) / torch.sum(weight)
    absrel_solver = AbsRel()
    abs_rel = absrel_solver.compute(gt=gt, pred=pred, weights=weight)

    # sq_rel = torch.mean(weight * (gt - pred) ** 2 / gt)
    sq_rel_solver = SqRel()
    sq_rel = sq_rel_solver.compute(gt=gt, pred=pred, weights=weight)

    log10 = torch.sum(weight * torch.abs(torch.log10(pred / gt))) / torch.sum(weight)

    return abs, abs_rel, sq_rel, rmse, rmse_log, log10, a11, a1, a2, a3


class SphericalEvaluator(object):

    """ Evaluating depth estimation considering weights

        Computing depth estimation metrics which weights each pixel by its position on spherical surface.

        Attributes:
            median_align: Do scale alignment or not by median depth.
            metrics: the evaluation results. See details in Pano3D paper and gitHub url:
             Albanis, Georgios, et al. "Pano3d: A holistic benchmark and a solid baseline for 360deg depth estimation."
             Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
             https://github.com/VCL3D/Pano3D
        """

    def __init__(self, median_align=False):
        self.median_align = median_align
        # Error and Accuracy metric trackers
        self.metrics = {}
        self.metrics["err/abs_"] = AverageMeter()
        self.metrics["err/abs_rel"] = AverageMeter()
        self.metrics["err/sq_rel"] = AverageMeter()
        self.metrics["err/rms"] = AverageMeter()
        self.metrics["err/log_rms"] = AverageMeter()
        self.metrics["err/log10"] = AverageMeter()
        self.metrics["acc/a1"] = AverageMeter()
        self.metrics["acc/a2"] = AverageMeter()
        self.metrics["acc/a3"] = AverageMeter()
        self.metrics["acc/a11"] = AverageMeter()
        self.metrics["acc/a1"] = AverageMeter()
        self.metrics["acc/a2"] = AverageMeter()
        self.metrics["acc/a3"] = AverageMeter()

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the model
        """
        self.metrics["err/abs_"].reset()
        self.metrics["err/abs_rel"].reset()
        self.metrics["err/sq_rel"].reset()
        self.metrics["err/rms"].reset()
        self.metrics["err/log_rms"].reset()
        self.metrics["err/log10"].reset()
        self.metrics["acc/a11"].reset()
        self.metrics["acc/a1"].reset()
        self.metrics["acc/a2"].reset()
        self.metrics["acc/a3"].reset()

    def compute_eval_metrics(self, gt_depth, pred_depth, mask=None, weight=None, min_depth=0.1, max_depth=10.0):
        """
        Computes metrics used to evaluate the model
        """
        N = gt_depth.shape[0]

        abs, abs_rel, sq_rel, rmse, rmse_log, log10, a11, a1, a2, a3 = \
            compute_depth_spherical_metrics(gt_depth, pred_depth, mask, self.median_align,
                                            min_depth=min_depth, max_depth=max_depth, weight=weight)

        self.metrics["err/abs_"].update(abs_, N)
        self.metrics["err/abs_rel"].update(abs_rel, N)
        self.metrics["err/sq_rel"].update(sq_rel, N)
        self.metrics["err/rms"].update(rms, N)
        self.metrics["err/log_rms"].update(rms_log, N)
        self.metrics["err/log10"].update(log10, N)
        self.metrics["acc/a11"].update(a11, N)
        self.metrics["acc/a1"].update(a1, N)
        self.metrics["acc/a2"].update(a2, N)
        self.metrics["acc/a3"].update(a3, N)

    def print(self, dir=None):
        avg_metrics = []
        avg_metrics.append(self.metrics["err/abs_"].avg)
        avg_metrics.append(self.metrics["err/abs_rel"].avg)
        avg_metrics.append(self.metrics["err/sq_rel"].avg)
        avg_metrics.append(self.metrics["err/rms"].avg)
        avg_metrics.append(self.metrics["err/log_rms"].avg)
        avg_metrics.append(self.metrics["err/log10"].avg)
        avg_metrics.append(self.metrics["acc/a11"].avg)
        avg_metrics.append(self.metrics["acc/a1"].avg)
        avg_metrics.append(self.metrics["acc/a2"].avg)
        avg_metrics.append(self.metrics["acc/a3"].avg)

        logger.info(
            "\n  " + ("{:>9} | " * 4).format("abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10",
                                             "a11", "a1", "a2", "a3")
            + "\n  " + ("&  {: 8.5f} " * 4).format(*avg_metrics))
        if dir is not None:
            file = os.path.join(dir, "result.txt")
            with open(file, 'w') as f:
                print("\n  " + ("{:>9} | " * 9).format("abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10",
                                                       "acc/a11", "acc/a1", "acc/a2", "acc/a3"), file=f)
                print(("&  {: 8.5f} " * 9).format(*avg_metrics), file=f)