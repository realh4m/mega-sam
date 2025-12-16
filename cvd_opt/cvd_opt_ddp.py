# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Consistent video depth optimization with DDP and Mini-batching."""

import argparse
import os
from pathlib import Path

from geometry_utils import NormalGenerator
import kornia
from lietorch import SE3
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint


def gradient_loss(gt, pred, u):
  """Gradient loss."""
  del u
  diff = pred - gt
  v_gradient = torch.abs(diff[..., 0:-2, 1:-1] - diff[..., 2:, 1:-1])
  h_gradient = torch.abs(diff[..., 1:-1, 0:-2] - diff[..., 1:-1, 2:])

  pred_grad = torch.abs(pred[..., 0:-2, 1:-1] - (pred[..., 2:, 1:-1])) + \
              torch.abs(pred[..., 1:-1, 0:-2] - pred[..., 1:-1, 2:])
  gt_grad = torch.abs(gt[..., 0:-2, 1:-1] - (gt[..., 2:, 1:-1])) + \
            torch.abs(gt[..., 1:-1, 0:-2] - gt[..., 1:-1, 2:])

  grad_diff = torch.abs(pred_grad - gt_grad)
  nearby_mask = (torch.exp(gt[..., 1:-1, 1:-1]) > 1.0).float().detach()
  weight = 1.0 - torch.exp(-(grad_diff * 5.0)).detach()
  weight *= nearby_mask

  g_loss = torch.mean(h_gradient * weight) + torch.mean(v_gradient * weight)
  return g_loss

def si_loss(gt, pred):
  """Scale-invariant loss."""
  log_gt = torch.log(torch.clamp(gt, 1e-3, 1e3)).view(gt.shape[0], -1)
  log_pred = torch.log(torch.clamp(pred, 1e-3, 1e3)).view(pred.shape[0], -1)
  log_diff = log_gt - log_pred
  num_pixels = gt.shape[-2] * gt.shape[-1]
  data_loss = torch.sum(log_diff**2, dim=-1) / num_pixels - \
              torch.sum(log_diff, dim=-1) ** 2 / (num_pixels**2)
  return torch.mean(data_loss)

def sobel_fg_alpha(disp, mode="sobel", beta=10.0):
  """Sobel foreground alpha."""
  sobel_grad = kornia.filters.spatial_gradient(disp, mode=mode, normalized=False)
  sobel_mag = torch.sqrt(
    sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2
  )
  alpha = torch.exp(-1.0 * beta * sobel_mag).detach()
  return alpha


ALPHA_MOTION = 0.25
RESIZE_FACTOR = 0.5
FLOW_BATCH_SIZE = 256


def consistency_loss(
    cam_c2w, K, K_inv, disp_data, init_disp, uncertainty,
    flows, flow_masks, ii, jj, compute_normals, fg_alpha, grid,
    w_ratio=1.0, w_flow=0.2, w_si=1.0, w_grad=2.0, w_normal=4.0,
    flow_batch_size=FLOW_BATCH_SIZE
):
  """Consistency loss with mini-batching."""
  device, dtype = disp_data.device, disp_data.dtype
  _, H, W = disp_data.shape

  cam_1to2 = torch.bmm(
    torch.linalg.inv(torch.index_select(cam_c2w, dim=0, index=jj)),
    torch.index_select(cam_c2w, dim=0, index=ii)
  )
  KK = torch.inverse(K_inv)

  def _flow_block(
      cam_1to2_batch,
      flows_batch,
      flow_masks_batch,
      ii_batch,
      jj_batch,
      disp_data_,
      init_disp_,
      uncertainty_,
  ):
    flows_step = flows_batch.permute(0, 2, 3, 1)
    flow_masks_step = flow_masks_batch.permute(0, 2, 3, 1).squeeze(-1)
    pixel_locations = grid + flows_step
    
    res_factor = torch.tensor([W - 1.0, H - 1.0], device=device)
    res_factor = res_factor[None, None, None, ...]
    norm_pixel_locs = 2 * (pixel_locations / res_factor) - 1.0

    disp_sampled = torch.nn.functional.grid_sample(
      torch.index_select(disp_data_, dim=0, index=jj_batch)[:, None, ...],
      norm_pixel_locs, align_corners=True
    )

    uu = torch.index_select(uncertainty_, dim=0, index=ii_batch).squeeze(1)
    grid_h = torch.cat([grid, torch.ones_like(grid[..., 0:1])], dim=-1).unsqueeze(-1)
    ref_depth = 1.0 / torch.clamp(
      torch.index_select(disp_data_, dim=0, index=ii_batch), 1e-3, 1e3
    )

    pts_3d_ref = ref_depth[..., None, None] * (K_inv[None, None, None] @ grid_h)
    pts_3d_tgt = (cam_1to2_batch[:, None, None, :3, :3] @ pts_3d_ref) + \
                 cam_1to2_batch[:, None, None, :3, 3:4]
    
    depth_tgt = pts_3d_tgt[:, :, :, 2:3, 0]
    disp_tgt = 1.0 / torch.clamp(depth_tgt, 0.1, 1e3)
    pts_2D_tgt = K[None, None, None] @ pts_3d_tgt

    m_step_ = flow_masks_step * (pts_2D_tgt[:, :, :, 2, 0] > 0.1)
    pts_2D_tgt = pts_2D_tgt[:, :, :, :2, 0] / \
                 torch.clamp(pts_2D_tgt[:, :, :, 2:, 0], 1e-3, 1e3)

    disp_sampled = torch.clamp(disp_sampled, 1e-3, 1e2)
    disp_tgt = torch.clamp(disp_tgt, 1e-3, 1e2)

    ratio = torch.maximum(
      disp_sampled.squeeze() / disp_tgt.squeeze(),
      disp_tgt.squeeze() / disp_sampled.squeeze()
    )
    ratio_error = torch.abs(ratio - 1.0)

    r_num = torch.sum((ratio_error * uu + ALPHA_MOTION * torch.log(1.0/uu)) * m_step_)
    r_den = torch.sum(m_step_) + 1e-8

    f_err = torch.abs(pts_2D_tgt - pixel_locations)
    f_num = torch.sum(
      (f_err * uu[..., None] + ALPHA_MOTION * torch.log(1.0/uu[..., None])) * \
      m_step_[..., None]
    )
    f_den = torch.sum(m_step_) * 2.0 + 1e-8
    return r_num, r_den, f_num, f_den

  def _surface_block(disp_data_, init_disp_, uncertainty_):
    disp_data_ds = disp_data_[:, None, ...]
    init_disp_ds = init_disp_[:, None, ...]
    K_rescale = KK.clone()
    K_inv_rescale = torch.inverse(K_rescale)
    
    pred_normal = compute_normals[0](
      1.0 / torch.clamp(disp_data_ds, 1e-3, 1e3), K_inv_rescale[None]
    )
    init_normal = compute_normals[0](
      1.0 / torch.clamp(init_disp_ds, 1e-3, 1e3), K_inv_rescale[None]
    )

    loss_normal = torch.mean(
        fg_alpha * (1.0 - torch.sum(pred_normal * init_normal, dim=1))
    )
    
    loss_grad = 0.0
    for scale in range(4):
      iv = 2**scale
      d_ds = torch.nn.functional.interpolate(
        disp_data_[:, None, ...], scale_factor=(1.0/iv, 1.0/iv), mode="nearest-exact"
      )
      i_ds = torch.nn.functional.interpolate(
        init_disp_[:, None, ...], scale_factor=(1.0/iv, 1.0/iv), mode="nearest-exact"
      )
      u_rs = torch.nn.functional.interpolate(
        uncertainty_, scale_factor=(1.0/iv, 1.0/iv), mode="nearest-exact"
      )
      loss_grad += gradient_loss(torch.log(d_ds), torch.log(i_ds), u_rs)
    return loss_normal, loss_grad

  rn_t, rd_t, fn_t, fd_t = [torch.zeros((), dtype=dtype, device=device) for _ in range(4)]
  for start in range(0, flows.shape[0], flow_batch_size):
    end = min(flows.shape[0], start + flow_batch_size)
    rn, rd, fn, fd = checkpoint(
      _flow_block, cam_1to2[start:end], flows[start:end], flow_masks[start:end],
      ii[start:end], jj[start:end], disp_data, init_disp, uncertainty,
      preserve_rng_state=False, use_reentrant=False
    )
    rn_t += rn; rd_t += rd; fn_t += fn; fd_t += fd

  loss_d_ratio = rn_t / (rd_t + 1e-8)
  loss_flow = fn_t / (fd_t + 1e-8)
  loss_prior = si_loss(init_disp, disp_data)
  
  loss_normal, loss_grad = checkpoint(
    _surface_block, disp_data, init_disp, uncertainty,
    preserve_rng_state=False, use_reentrant=False
  )

  return (w_ratio * loss_d_ratio + w_si * loss_prior + w_flow * loss_flow + 
          w_normal * loss_normal + loss_grad * w_grad)

class CVDOptimizer(torch.nn.Module):
  def __init__(self, disp_data, uncertainty, poses_th, K, K_inv, init_disp, compute_normals, fg_alpha):
    super().__init__()
    self.disp_data = torch.nn.Parameter(disp_data.clone())
    self.uncertainty = torch.nn.Parameter(uncertainty.clone())
    self.register_buffer("poses_th", poses_th)
    self.register_buffer("K", K)
    self.register_buffer("K_inv", K_inv)
    self.register_buffer("init_disp", init_disp)
    self.register_buffer("fg_alpha", fg_alpha)
    
    _, H, W = disp_data.shape
    xx = torch.arange(0, W, device=disp_data.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=disp_data.device).view(-1, 1).repeat(1, W)
    grid = torch.cat((xx.view(1, 1, H, W), yy.view(1, 1, H, W)), 1).float()
    self.register_buffer("grid", grid.permute(0, 2, 3, 1))
    self.compute_normals = compute_normals

  def forward(self, flows, flow_masks, ii, jj, **kwargs):
    cam_c2w = SE3(self.poses_th).inv().matrix()
    return consistency_loss(
      cam_c2w, self.K, self.K_inv, torch.clamp(self.disp_data, 1e-3, 1e3),
      self.init_disp, torch.clamp(self.uncertainty, 1e-4, 1e3),
      flows, flow_masks, ii, jj, self.compute_normals, self.fg_alpha, self.grid, **kwargs
    )

def setup_ddp(rank, world_size):
  os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "12355"
  dist.init_process_group("nccl", rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)

def train_worker(rank, world_size, args):
  setup_ddp(rank, world_size)
  device = torch.device(f"cuda:{rank}")
  root, cache, scene = os.getcwd() + "/reconstructions", "./cache_flow", args.scene_name
  
  if rank == 0: print("***************************** ", scene)
  
  img_data = np.load(os.path.join(root, scene, "images.npy"))[:, ::-1, ...]
  disp_np = np.load(os.path.join(root, scene.replace("_opt", ""), "disps.npy")) + 1e-6
  intrinsics = np.load(os.path.join(root, scene, "intrinsics.npy"))[0]
  poses = np.load(os.path.join(root, scene, "poses.npy"))
  mot_prob = np.load(os.path.join(root, scene, "motion_prob.npy"))
  flows = np.load(f"{cache}/{scene}/flows.npy", allow_pickle=True)
  masks = np.float32(np.load(f"{cache}/{scene}/flows_masks.npy", allow_pickle=True))
  iijj = np.load(f"{cache}/{scene}/ii-jj.npy", allow_pickle=True)

  t_pairs = len(iijj[0]); p_gpu = t_pairs // world_size; s_idx = rank * p_gpu
  e_idx = s_idx + p_gpu if rank < world_size - 1 else t_pairs

  l_flows = torch.from_numpy(np.ascontiguousarray(flows[s_idx:e_idx])).float().to(device)
  l_masks = torch.from_numpy(np.ascontiguousarray(masks[s_idx:e_idx])).float().to(device)
  l_ii = torch.from_numpy(np.ascontiguousarray(iijj[0, s_idx:e_idx])).long().to(device)
  l_jj = torch.from_numpy(np.ascontiguousarray(iijj[1, s_idx:e_idx])).long().to(device)

  K = torch.eye(3, device=device).float()
  intr_th = torch.from_numpy(intrinsics).to(device)
  K[0, 0], K[1, 1], K[0, 2], K[1, 2] = intr_th[0], intr_th[1], intr_th[2], intr_th[3]
  K_o = K.clone(); K[0:2, ...] *= RESIZE_FACTOR; K_inv = torch.linalg.inv(K)

  img_data_pt = torch.from_numpy(np.ascontiguousarray(img_data)).float().to(device) / 255.0
  init_disp = torch.from_numpy(disp_np).float().to(device)
  init_disp = torch.nn.functional.interpolate(
    init_disp.unsqueeze(1), scale_factor=RESIZE_FACTOR, mode="bilinear"
  ).squeeze(1)
  
  fa = (sobel_fg_alpha(init_disp[:, None, ...]) > 0.2).squeeze(1).float() + 0.2
  unc = torch.nn.functional.interpolate(
    torch.from_numpy(mot_prob).unsqueeze(1).to(device), scale_factor=4, mode="bilinear"
  )
  unc = torch.clamp(unc, 1e-3, 0.5)
  
  comp_normals = [NormalGenerator(init_disp.shape[-2], init_disp.shape[-1])]
  model = DDP(CVDOptimizer(init_disp, unc, torch.as_tensor(poses, device=device).float(), 
                           K, K_inv, init_disp, comp_normals, fa), device_ids=[rank], find_unused_parameters=True)

  ls_ = torch.log(torch.ones(init_disp.shape[0], device=device))
  sh_ = torch.zeros(init_disp.shape[0], device=device)
  ls_.requires_grad = sh_.requires_grad = True

  opt = torch.optim.Adam([{"params": ls_, "lr": 1e-2}, {"params": sh_, "lr": 1e-2}, 
                          {"params": model.parameters(), "lr": 1e-2}])

  for i in range(100):
    opt.zero_grad()
    sc_ = torch.exp(ls_)[..., None, None]
    scaled_d = torch.clamp(model.module.disp_data * sc_ + sh_[..., None, None], 1e-3, 1e3)
    loss = consistency_loss(SE3(model.module.poses_th).inv().matrix(), K, K_inv, scaled_d, 
                            model.module.init_disp, torch.clamp(model.module.uncertainty, 1e-4, 1e3), 
                            l_flows, l_masks, l_ii, l_jj, comp_normals, fa, model.module.grid)
    loss.backward()
    ls_.grad, sh_.grad = torch.nan_to_num(ls_.grad), torch.nan_to_num(sh_.grad)
    opt.step()
    if rank == 0: print(f"step {i}, loss: {loss.item()}")

  with torch.inference_mode():
    model.module.disp_data.data = torch.clamp(
      model.module.disp_data * torch.exp(ls_)[..., None, None].detach() + sh_[..., None, None].detach(), 1e-3, 1e3
    )
    model.module.init_disp.data = model.module.disp_data.data.clone()

  opt = torch.optim.Adam([{"params": model.module.disp_data, "lr": 5e-3}, 
                          {"params": model.module.uncertainty, "lr": 5e-3}])

  for i in range(400):
    opt.zero_grad()
    loss = model(l_flows, l_masks, l_ii, l_jj, w_grad=args.w_grad, w_normal=args.w_normal)
    loss.backward()
    for p in model.parameters():
      if p.grad is not None: p.grad = torch.nan_to_num(p.grad, nan=0.0)
    opt.step()
    if rank == 0: print(f"step {i}, loss: {loss.item()}")

  if rank == 0:
    opt_np = torch.nn.functional.interpolate(
      model.module.disp_data.unsqueeze(1), scale_factor=2, mode="bilinear"
    ).squeeze(1).detach().cpu().numpy()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    np.savez(f"{args.output_dir}/{scene}_sgd_cvd_hr_ddp.npz", 
             images=np.uint8(img_data_pt.cpu().numpy().transpose(0, 2, 3, 1) * 255.0), 
             depths=np.clip(np.float16(1.0/opt_np), 1e-3, 1e2), 
             intrinsic=K_o.cpu().numpy(), 
             cam_c2w=SE3(model.module.poses_th).inv().matrix().detach().cpu().numpy())
    print(f"Results saved to {args.output_dir}/{scene}_sgd_cvd_hr_ddp.npz")

  dist.destroy_process_group()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--w_grad", type=float, default=2.0)
  parser.add_argument("--w_normal", type=float, default=6.0)
  parser.add_argument("--output_dir", type=str, default="outputs_cvd")
  parser.add_argument("--scene_name", type=str); parser.add_argument("--world_size", type=int, default=2)
  args = parser.parse_args()
  mp.spawn(train_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
