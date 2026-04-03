import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.pcae import PointCloudAccuracyEstimator
import utils.helpers as helpers


class CAPromptGenerator(nn.Module):
    """
    Consistency-Aware Prompt Generator (CAPG)

    Implements the three-stage bbox generation described in the paper (Sec. 2.1):
      1) 1D Projection  — project scribbles S and coarse predictions Y onto x-/y-axes (Eq. 1)
      2) Consistency Computation — compute D̄_pos/neg, A_pos/neg, CS per axis (Eq. 2–4)
      3) Consistency Prompt Generation — derive bp, fuse with y-projection, expand by δ_slack (Eq. 5–8)

    Point sampling via PCU is unchanged.
    """

    # Fixed pixel counts from the paper (Sec. 2.1, Eq. 2)
    NS_GT0 = 60   # N_{s>0}: pixels inside enriched scribbles used for D̄_pos
    NS_EQ0 = 10   # N_{s=0}: pixels outside enriched scribbles used for D̄_neg and bp extent

    def __init__(self, n: int, not_n: int, ignore_index: int = 0):
        super().__init__()
        self.n = n
        self.not_n = not_n
        self.ignore_index = ignore_index
        self.pcu = PointCloudAccuracyEstimator()

    # ------------------------------------------------------------------
    # Stage 1 — 1D Projection (Eq. 1)
    # ------------------------------------------------------------------
    @staticmethod
    def _project_1d(mask: np.ndarray):
        proj_x = mask.max(axis=0).astype(np.float32)   # (W,)
        proj_y = mask.max(axis=1).astype(np.float32)   # (H,)
        return proj_x, proj_y

    # ------------------------------------------------------------------
    # Stage 2 — Consistency Computation (Eq. 2–4)
    # ------------------------------------------------------------------
    @staticmethod
    def _consistency_score(s_proj: np.ndarray, y_proj: np.ndarray, ns_gt0: int, ns_eq0: int) -> float:
        pos_mask = s_proj > 0
        neg_mask = ~pos_mask

        d_pos = float(((1.0 - y_proj[pos_mask]) ** 2).sum()) / max(ns_gt0, 1) if pos_mask.any() else 0.0
        d_neg = float((y_proj[neg_mask] ** 2).sum())          / max(ns_eq0, 1) if neg_mask.any() else 0.0

        a_pos = 1.0 / (1.0 + d_pos)
        a_neg = 1.0 / (1.0 + d_neg)
        return min(a_pos, a_neg)

    # ------------------------------------------------------------------
    # Stage 3 — Consistency Prompt Generation (Eq. 5–8)
    # ------------------------------------------------------------------
    @staticmethod
    def _expand_proj(s_proj: np.ndarray, bp_extent: int, length: int) -> np.ndarray:
        """
        Eq. 5 正确语义：
        bp_extent = ⌊CS · N_{s=0}⌋ 是 scribble 投影边界的外扩半径（像素数）。
        取 scribble 投影的非零区间 [lo, hi]，向外各扩 bp_extent 列/行，
        得到 bp 二值投影向量。
        当 scribble 投影为空时返回全零向量（由调用方兜底）。
        """
        bp = np.zeros(length, dtype=np.float32)
        nonzero = np.where(s_proj > 0)[0]
        if len(nonzero) == 0:
            return bp
        lo = max(0,          nonzero.min() - bp_extent)
        hi = min(length - 1, nonzero.max() + bp_extent)
        bp[lo: hi + 1] = 1.0
        return bp

    def _bboxes_from_projection_consistency(
        self,
        scribbles_np: np.ndarray,   # (B, C, H, W) per-class one-hot scribble
        pred_oh_np:   np.ndarray,   # (B, C, H, W) argmax one-hot prediction
        probs_np:     np.ndarray,   # (B, C, H, W) softmax probabilities（兜底用）
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

                S      = scribbles_np[b, c].astype(np.float32)
                Y      = pred_oh_np[b, c].astype(np.float32)
                Pprob  = probs_np[b, c]                          # (H, W) float

                # Stage 1 — 1D projections
                s_x, s_y = self._project_1d(S)
                y_x, y_y = self._project_1d(Y)

                # Stage 2 — consistency scores
                cs_x = self._consistency_score(s_x, y_x, self.NS_GT0, self.NS_EQ0)
                cs_y = self._consistency_score(s_y, y_y, self.NS_GT0, self.NS_EQ0)

                # Stage 3a — bp：scribble 投影边界外扩 bp_extent 个像素 (Eq. 5 正确语义)
                bp_x_extent = int(cs_x * self.NS_EQ0)
                bp_y_extent = int(cs_y * self.NS_EQ0)
                bp_x = self._expand_proj(s_x, bp_x_extent, W)
                bp_y = self._expand_proj(s_y, bp_y_extent, H)

                # Stage 3b — OR fusion with coarse prediction (Eq. 6)
                p_x = np.clip(bp_x + y_x, 0.0, 1.0)
                p_y = np.clip(bp_y + y_y, 0.0, 1.0)

                # 兜底 1：scribble 为空但预测非空，直接用预测投影
                if not p_x.any():
                    p_x = y_x.copy()
                if not p_y.any():
                    p_y = y_y.copy()

                # 兜底 2：预测也全空，用 softmax 概率投影（max over rows/cols）
                if not p_x.any() or not p_y.any():
                    prob_x = Pprob.max(axis=0)   # (W,)
                    prob_y = Pprob.max(axis=1)   # (H,)
                    tau = float(np.percentile(Pprob, 90))
                    if not p_x.any():
                        p_x = (prob_x >= tau).astype(np.float32)
                    if not p_y.any():
                        p_y = (prob_y >= tau).astype(np.float32)

                # 仍为空 → 零框
                if not p_x.any() or not p_y.any():
                    continue

                # Stage 3c — min/max + slack (Eq. 7–8)
                x_nonzero = np.where(p_x > 0)[0]
                y_nonzero = np.where(p_y > 0)[0]
                x_min = int(max(0,     x_nonzero.min() - delta_slack))
                x_max = int(min(W - 1, x_nonzero.max() + delta_slack))
                y_min = int(max(0,     y_nonzero.min() - delta_slack))
                y_max = int(min(H - 1, y_nonzero.max() + delta_slack))

                # 与 2D scribble 直接 bbox 取并集（union hull）：
                # 保证 1D projection 框不比"scribble 直接外接框 + slack"更小，
                # 防止稀疏 scribble 场景下框过紧导致 MedSAM 漏分割。
                scr_ys, scr_xs = np.where(S > 0)
                if len(scr_xs) > 0:
                    sx_min = int(max(0,     scr_xs.min() - delta_slack))
                    sx_max = int(min(W - 1, scr_xs.max() + delta_slack))
                    sy_min = int(max(0,     scr_ys.min() - delta_slack))
                    sy_max = int(min(H - 1, scr_ys.max() + delta_slack))
                    x_min = min(x_min, sx_min)
                    x_max = max(x_max, sx_max)
                    y_min = min(y_min, sy_min)
                    y_max = max(y_max, sy_max)

                bboxes[b, c] = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        return bboxes   # (B, C, 4)

    # ------------------------------------------------------------------
    # Unchanged helpers (PCU interface, point sampling)
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_choice(arr: np.ndarray, k: int):
        """Sample k rows from arr; sample with replacement if arr is smaller than k."""
        if len(arr) == 0:
            return arr
        replace = len(arr) < k
        idx = np.random.choice(len(arr), k, replace=replace)
        return arr[idx]

    def _get_points(self, coords: np.ndarray, n: int):
        if n <= 0:
            return np.empty((0, 2), dtype=np.int64)
        return self._safe_choice(coords, n)

    def _reduce_mask(self, binary_mask: np.ndarray, background_mask: np.ndarray, k_pos: int, k_bg: int):
        """
        Sample k_pos positive points from binary_mask and k_bg negative points from background_mask.

        Returns:
            coords: (k_pos+k_bg, 2) — (y, x)
            labels: (k_pos+k_bg,)   — 1/0
        """
        fg_ys, fg_xs = np.where(binary_mask > 0)
        bg_ys, bg_xs = np.where(background_mask > 0)
        fg_coords = np.stack([fg_ys, fg_xs], axis=1) if len(fg_ys) else np.empty((0, 2), dtype=np.int64)
        bg_coords = np.stack([bg_ys, bg_xs], axis=1) if len(bg_ys) else np.empty((0, 2), dtype=np.int64)
        pos_pts = self._get_points(fg_coords, max(k_pos, 0))
        neg_pts = self._get_points(bg_coords, max(k_bg, 0))
        coords = (
            np.concatenate([pos_pts, neg_pts], axis=0)
            if (len(pos_pts) or len(neg_pts))
            else np.empty((0, 2), dtype=np.int64)
        )
        labels = (
            np.concatenate(
                [np.ones(len(pos_pts), dtype=np.int64), np.zeros(len(neg_pts), dtype=np.int64)], axis=0
            )
            if (len(pos_pts) or len(neg_pts))
            else np.empty((0,), dtype=np.int64)
        )
        return coords, labels

    def forward_pcu(self, labels_4d_torch: torch.Tensor, out_onehot_np: np.ndarray):
        """Preserved PCU interface — unchanged."""
        return self.pcu(labels_4d_torch, out_onehot_np)

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------
    def forward(self, scribbles: torch.Tensor, outputs: torch.Tensor):
        """
        Args:
            scribbles: (B, H, W) or (B, 1, H, W)
                - Binary/sparse mask {0,1}: broadcast to every class channel.
                - Class-ID label map (values 0..C-1, 255 ignored): converted to per-class one-hot.
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
            # Binary/sparse mask — broadcast to all class channels
            scribbles_t = s[:, None, :, :].repeat(1, C, 1, 1)
        else:
            # Class-ID map — convert to one-hot
            s = torch.clamp(s, 0, C - 1)
            scribbles_t = F.one_hot(s, num_classes=C).permute(0, 3, 1, 2).contiguous()

        # --- Numpy arrays for PCU and bbox ---
        outputs_oh  = helpers.to_onehot(outputs).detach().cpu().numpy()         # (B,C,H,W) PCU one-hot
        pred_lbl    = outputs.argmax(dim=1)                                      # (B,H,W)
        pred_oh_np  = (
            F.one_hot(pred_lbl, num_classes=C)
            .permute(0, 3, 1, 2)
            .contiguous()
            .detach().cpu().numpy()
        )                                                                         # (B,C,H,W) strict argmax

        scribbles_np    = scribbles_t.detach().cpu().numpy()                     # (B,C,H,W)
        scribbles_cpu_t = torch.from_numpy(scribbles_np)

        # softmax 概率（兜底用）
        probs_np = F.softmax(outputs, dim=1).detach().cpu().numpy()              # (B,C,H,W)

        # --- PCU ---
        pcu = self.forward_pcu(scribbles_cpu_t, outputs_oh)

        # --- Point sampling (unchanged) ---
        points_list, labels_list = [], []
        need = self.n + self.not_n
        for b in range(B):
            img_prompt = np.zeros((C, need, 2), dtype=np.int64)
            img_labels = np.zeros((C, need),    dtype=np.int64)

            for c in range(C):
                pcu_bc = float(pcu[b][c].item()) if hasattr(pcu, "shape") else float(pcu[b][c])
                num_pred      = int(round(pcu_bc       * self.n,     0))
                num_scribble  = self.n     - num_pred
                num_not_pred  = int(round((1 - pcu_bc) * self.not_n, 0))
                num_not_scrib = self.not_n - num_not_pred

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

        points = torch.stack(points_list, dim=0)   # (B, C, N, 2)
        labels = torch.stack(labels_list, dim=0)   # (B, C, N)

        # --- Bounding boxes via paper-faithful 1D projection consistency (Eq. 1–8) ---
        bboxes = self._bboxes_from_projection_consistency(
            scribbles_np=scribbles_np,
            pred_oh_np=pred_oh_np,
            probs_np=probs_np,
            H=H, W=W,
            delta_slack=10,
        )   # (B, C, 4) float32 on CPU

        points = points.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)

        return points, labels, bboxes