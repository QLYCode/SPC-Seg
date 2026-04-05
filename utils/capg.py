import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

from utils.pcae import PointCloudAccuracyEstimator
import utils.helpers as helpers


class CAPromptGenerator(nn.Module):
    """
    Consistency-Aware Prompt Generator (CAPG)

    Fixes vs. original:
      [F1] _consistency_score: normalise by actual pixel count (capped at NS_*)
           instead of fixed NS_*. Prevents d_pos/neg being artificially suppressed
           when the projection has fewer than NS_* active pixels.
      [F2] bp_extent = floor((1 - CS) * NS_EQ0):  HIGH consistency → small
           expansion (trust the scribble boundary); LOW consistency → large
           expansion (uncertain, be generous). Original had the sign inverted.
      [F3] bp seed: if scribble is non-empty but bp is still zero after expansion
           (happens when CS≈0 and extent=0), seed bp directly from s_proj.
      [F4] Soft hull replaces unconditional union hull: only expand the
           projection box to cover scribble pixels when the projection box would
           otherwise *exclude* them. Never expand unconditionally — that was
           inflating boxes on sparse scribbles and flooding MedSAM with background.
      [F5] PCU soft-clamp [0.15, 0.85]: prevents degenerate all-pred or
           all-scribble sampling at the extremes.
      [F6] Distance-transform-weighted positive point sampling: prefer points
           near the mask skeleton (far from boundary) — more stable SAM prompts.
    """

    NS_GT0   = 60
    NS_EQ0   = 10
    PCU_LO   = 0.15
    PCU_HI   = 0.85

    def __init__(self, n: int, not_n: int, ignore_index: int = 0):
        super().__init__()
        self.n = n
        self.not_n = not_n
        self.ignore_index = ignore_index
        self.pcu = PointCloudAccuracyEstimator()

    # ------------------------------------------------------------------
    # Stage 1 — 1D Projection (Eq. 1)  [unchanged]
    # ------------------------------------------------------------------
    @staticmethod
    def _project_1d(mask: np.ndarray):
        proj_x = mask.max(axis=0).astype(np.float32)   # (W,)
        proj_y = mask.max(axis=1).astype(np.float32)   # (H,)
        return proj_x, proj_y

    # ------------------------------------------------------------------
    # Stage 2 — Consistency Computation (Eq. 2–4)
    # [F1] use actual pixel counts for normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _consistency_score(
        s_proj: np.ndarray,
        y_proj: np.ndarray,
        ns_gt0: int,
        ns_eq0: int,
    ) -> float:
        pos_mask = s_proj > 0
        neg_mask = ~pos_mask

        # [F1] cap at paper constant but don't go below actual count
        norm_pos = max(min(int(pos_mask.sum()), ns_gt0), 1)
        norm_neg = max(min(int(neg_mask.sum()), ns_eq0), 1)

        d_pos = float(((1.0 - y_proj[pos_mask]) ** 2).sum()) / norm_pos if pos_mask.any() else 0.0
        d_neg = float((y_proj[neg_mask] ** 2).sum())          / norm_neg if neg_mask.any() else 0.0

        a_pos = 1.0 / (1.0 + d_pos)
        a_neg = 1.0 / (1.0 + d_neg)
        return min(a_pos, a_neg)

    # ------------------------------------------------------------------
    # Stage 3a — bp expansion
    # [F2] inverted extent: high CS → small expansion (trust boundary)
    # ------------------------------------------------------------------
    @staticmethod
    def _expand_proj(s_proj: np.ndarray, bp_extent: int, length: int) -> np.ndarray:
        bp = np.zeros(length, dtype=np.float32)
        nonzero = np.where(s_proj > 0)[0]
        if len(nonzero) == 0:
            return bp
        lo = max(0,          nonzero.min() - bp_extent)
        hi = min(length - 1, nonzero.max() + bp_extent)
        bp[lo: hi + 1] = 1.0
        return bp

    # ------------------------------------------------------------------
    # Bbox generation
    # ------------------------------------------------------------------
    def _bboxes_from_projection_consistency(
        self,
        scribbles_np: np.ndarray,
        pred_oh_np:   np.ndarray,
        probs_np:     np.ndarray,
        H: int,
        W: int,
        delta_slack: int = 10,
    ) -> torch.Tensor:
        B, C = scribbles_np.shape[:2]
        bboxes = torch.zeros((B, C, 4), dtype=torch.float32)

        for b in range(B):
            for c in range(C):
                if c == self.ignore_index:
                    continue

                S     = scribbles_np[b, c].astype(np.float32)
                Y     = pred_oh_np[b, c].astype(np.float32)
                Pprob = probs_np[b, c]                        # (H, W)

                # Stage 1
                s_x, s_y = self._project_1d(S)
                y_x, y_y = self._project_1d(Y)

                # Stage 2
                cs_x = self._consistency_score(s_x, y_x, self.NS_GT0, self.NS_EQ0)
                cs_y = self._consistency_score(s_y, y_y, self.NS_GT0, self.NS_EQ0)

                # Stage 3a — [F2] (1-CS)*NS_EQ0: high CS → tight boundary
                bp_x_extent = int((1.0 - cs_x) * self.NS_EQ0)
                bp_y_extent = int((1.0 - cs_y) * self.NS_EQ0)
                bp_x = self._expand_proj(s_x, bp_x_extent, W)
                bp_y = self._expand_proj(s_y, bp_y_extent, H)

                # [F3] seed bp from s_proj when scribble exists but bp still empty
                if not bp_x.any() and s_x.any():
                    bp_x = s_x.copy()
                if not bp_y.any() and s_y.any():
                    bp_y = s_y.copy()

                # Stage 3b — OR fusion (Eq. 6)
                p_x = np.clip(bp_x + y_x, 0.0, 1.0)
                p_y = np.clip(bp_y + y_y, 0.0, 1.0)

                # Fallback 1: scribble absent, prediction present
                if not p_x.any():
                    p_x = y_x.copy()
                if not p_y.any():
                    p_y = y_y.copy()

                # Fallback 2: both empty → softmax probability projection
                if not p_x.any() or not p_y.any():
                    prob_x = Pprob.max(axis=0)
                    prob_y = Pprob.max(axis=1)
                    tau = float(np.percentile(Pprob, 75))   # 75th (was 90th, too aggressive)
                    if not p_x.any():
                        p_x = (prob_x >= tau).astype(np.float32)
                    if not p_y.any():
                        p_y = (prob_y >= tau).astype(np.float32)

                if not p_x.any() or not p_y.any():
                    continue

                # Stage 3c — min/max + slack (Eq. 7–8)
                x_nz = np.where(p_x > 0)[0]
                y_nz = np.where(p_y > 0)[0]
                x_min = int(max(0,     x_nz.min() - delta_slack))
                x_max = int(min(W - 1, x_nz.max() + delta_slack))
                y_min = int(max(0,     y_nz.min() - delta_slack))
                y_max = int(min(H - 1, y_nz.max() + delta_slack))

                # [F4] soft hull: only expand when projection box clips scribble pixels
                # (prevents silent under-coverage without inflating the box generally)
                scr_ys, scr_xs = np.where(S > 0)
                if len(scr_xs) > 0:
                    sx_min = int(max(0,     scr_xs.min() - delta_slack))
                    sx_max = int(min(W - 1, scr_xs.max() + delta_slack))
                    sy_min = int(max(0,     scr_ys.min() - delta_slack))
                    sy_max = int(min(H - 1, scr_ys.max() + delta_slack))
                    if x_min > sx_min or x_max < sx_max:
                        x_min = min(x_min, sx_min)
                        x_max = max(x_max, sx_max)
                    if y_min > sy_min or y_max < sy_max:
                        y_min = min(y_min, sy_min)
                        y_max = max(y_max, sy_max)

                bboxes[b, c] = torch.tensor(
                    [x_min, y_min, x_max, y_max], dtype=torch.float32
                )

        return bboxes   # (B, C, 4)

    # ------------------------------------------------------------------
    # Point sampling helpers
    # [F6] distance-transform-weighted positive sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_choice(arr: np.ndarray, k: int):
        if len(arr) == 0:
            return arr
        replace = len(arr) < k
        idx = np.random.choice(len(arr), k, replace=replace)
        return arr[idx]

    @staticmethod
    def _weighted_choice(coords: np.ndarray, k: int, mask: np.ndarray) -> np.ndarray:
        """
        Sample k points from coords, weighted by distance-transform of mask.
        Falls back to uniform sampling when mask is too small to bother.
        """
        if len(coords) == 0:
            return coords
        if len(coords) < k * 2:
            # Too few candidates — uniform is fine
            replace = len(coords) < k
            return coords[np.random.choice(len(coords), k, replace=replace)]

        # Distance transform on the binary mask
        dist = distance_transform_edt(mask)
        w = dist[coords[:, 0], coords[:, 1]].astype(np.float64)
        w_sum = w.sum()
        if w_sum < 1e-8:
            replace = len(coords) < k
            return coords[np.random.choice(len(coords), k, replace=replace)]
        w /= w_sum
        idx = np.random.choice(len(coords), k, replace=False, p=w)
        return coords[idx]

    def _get_points(self, coords: np.ndarray, n: int):
        if n <= 0:
            return np.empty((0, 2), dtype=np.int64)
        return self._safe_choice(coords, n)

    def _reduce_mask(
        self,
        binary_mask: np.ndarray,
        background_mask: np.ndarray,
        k_pos: int,
        k_bg: int,
    ):
        fg_ys, fg_xs = np.where(binary_mask > 0)
        bg_ys, bg_xs = np.where(background_mask > 0)
        fg_coords = np.stack([fg_ys, fg_xs], axis=1) if len(fg_ys) else np.empty((0, 2), dtype=np.int64)
        bg_coords = np.stack([bg_ys, bg_xs], axis=1) if len(bg_ys) else np.empty((0, 2), dtype=np.int64)

        # [F6] distance-weighted positive sampling; uniform for negatives
        if k_pos > 0 and len(fg_coords) > 0:
            pos_pts = self._weighted_choice(fg_coords, max(k_pos, 0), binary_mask)
        else:
            pos_pts = self._get_points(fg_coords, max(k_pos, 0))

        neg_pts = self._get_points(bg_coords, max(k_bg, 0))

        coords = (
            np.concatenate([pos_pts, neg_pts], axis=0)
            if (len(pos_pts) or len(neg_pts))
            else np.empty((0, 2), dtype=np.int64)
        )
        labels = (
            np.concatenate(
                [np.ones(len(pos_pts), dtype=np.int64),
                 np.zeros(len(neg_pts), dtype=np.int64)],
                axis=0,
            )
            if (len(pos_pts) or len(neg_pts))
            else np.empty((0,), dtype=np.int64)
        )
        return coords, labels

    def forward_pcu(self, labels_4d_torch: torch.Tensor, out_onehot_np: np.ndarray):
        return self.pcu(labels_4d_torch, out_onehot_np)

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------
    def forward(self, scribbles: torch.Tensor, outputs: torch.Tensor):
        """
        Args:
            scribbles: (B, H, W) or (B, 1, H, W)
            outputs:   (B, C, H, W) logits or probabilities.

        Returns:
            points: (B, C, N, 2)  — (x, y) prompt coordinates
            labels: (B, C, N)     — 1 = positive, 0 = negative
            bboxes: (B, C, 4)     — [x_min, y_min, x_max, y_max]
        """
        assert outputs.ndim == 4, f"outputs must be (B,C,H,W), got {outputs.shape}"
        device = outputs.device
        B, C, H, W = outputs.shape

        # --- Normalise scribbles to per-class one-hot (B, C, H, W) ---
        if scribbles.ndim == 4 and scribbles.shape[1] == 1:
            s = scribbles[:, 0]
        elif scribbles.ndim == 3:
            s = scribbles
        else:
            raise ValueError(f"scribbles shape invalid: {scribbles.shape}")

        s = s.long()
        ignore_mask_255 = s == 255
        if ignore_mask_255.any():
            s = torch.where(ignore_mask_255, torch.zeros_like(s), s)

        if s.max().item() <= 1 and s.min().item() >= 0:
            scribbles_t = s[:, None, :, :].repeat(1, C, 1, 1)
        else:
            s = torch.clamp(s, 0, C - 1)
            scribbles_t = F.one_hot(s, num_classes=C).permute(0, 3, 1, 2).contiguous()

        # --- Numpy arrays ---
        outputs_oh = helpers.to_onehot(outputs).detach().cpu().numpy()
        pred_lbl   = outputs.argmax(dim=1)
        pred_oh_np = (
            F.one_hot(pred_lbl, num_classes=C)
            .permute(0, 3, 1, 2)
            .contiguous()
            .detach().cpu().numpy()
        )
        scribbles_np    = scribbles_t.detach().cpu().numpy()
        scribbles_cpu_t = torch.from_numpy(scribbles_np)
        probs_np        = F.softmax(outputs, dim=1).detach().cpu().numpy()

        # --- PCU ---
        pcu = self.forward_pcu(scribbles_cpu_t, outputs_oh)

        # --- Point sampling ---
        points_list, labels_list = [], []
        need = self.n + self.not_n
        for b in range(B):
            img_prompt = np.zeros((C, need, 2), dtype=np.int64)
            img_labels = np.zeros((C, need),    dtype=np.int64)

            for c in range(C):
                pcu_bc = float(pcu[b][c].item()) if hasattr(pcu, "shape") else float(pcu[b][c])
                # [F5] soft-clamp PCU to avoid degenerate all-one-source sampling
                pcu_bc = float(np.clip(pcu_bc, self.PCU_LO, self.PCU_HI))

                num_pred      = int(round(pcu_bc         * self.n,     0))
                num_scribble  = self.n       - num_pred
                num_not_pred  = int(round((1.0 - pcu_bc) * self.not_n, 0))
                num_not_scrib = self.not_n   - num_not_pred

                scrib_c = scribbles_np[b, c]
                scrib_0 = scribbles_np[b, 0]
                out_c   = outputs_oh[b, c]
                out_0   = outputs_oh[b, 0]

                pts_s, lbs_s = self._reduce_mask(scrib_c, scrib_0, num_scribble, num_not_pred)
                pts_p, lbs_p = self._reduce_mask(out_c,   out_0,   num_pred,     num_not_scrib)

                prompt_xy = np.concatenate([pts_s, pts_p], axis=0)
                prompt_xy = prompt_xy[:, [1, 0]] if prompt_xy.size else np.zeros((0, 2), dtype=np.int64)
                prompt_lb = np.concatenate([lbs_s, lbs_p], axis=0)

                if prompt_xy.shape[0] < need:
                    pad = need - prompt_xy.shape[0]
                    if prompt_xy.shape[0] > 0:
                        pad_pts = np.tile(prompt_xy[-1:], (pad, 1))
                        pad_lb  = np.tile(prompt_lb[-1:],  (pad,))
                    else:
                        pad_pts = np.zeros((pad, 2), dtype=np.int64)
                        pad_lb  = np.zeros((pad,),   dtype=np.int64)
                    prompt_xy = np.concatenate([prompt_xy, pad_pts], axis=0)
                    prompt_lb = np.concatenate([prompt_lb,  pad_lb], axis=0)

                img_prompt[c] = prompt_xy[:need]
                img_labels[c] = prompt_lb[:need]

            points_list.append(torch.as_tensor(img_prompt, dtype=torch.long))
            labels_list.append(torch.as_tensor(img_labels, dtype=torch.long))

        points = torch.stack(points_list, dim=0)
        labels = torch.stack(labels_list, dim=0)

        # --- Bboxes ---
        bboxes = self._bboxes_from_projection_consistency(
            scribbles_np=scribbles_np,
            pred_oh_np=pred_oh_np,
            probs_np=probs_np,
            H=H, W=W,
            delta_slack=10,
        )

        return points.to(device), labels.to(device), bboxes.to(device)