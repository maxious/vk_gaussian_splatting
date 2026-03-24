import torch
import torch.nn.functional as F
from gsplat import project_gaussians, rasterize_gaussians_simple_loop


class GaussianRenderer(torch.autograd.Function):
    @staticmethod
    def render(
        xyz,
        feature,
        scale,
        rotation,
        opacity,
        test_c2w,
        test_intr,
        W,
        H,
        sh_degree,
        near_plane,
        far_plane,
    ):
        """Render Gaussians using gsplat-pytorch (pure PyTorch, works on XPU)."""
        opacity = opacity.sigmoid().squeeze(-1)  # (N,)
        scale = scale.exp()  # Convert log-scale to scale
        rotation = F.normalize(rotation, p=2, dim=-1)  # (N, 4)

        # Invert camera-to-world to get world-to-camera (view matrix)
        try:
            test_w2c = test_c2w.inverse()
        except:
            eps = 1e-6
            c2w_stable = test_c2w + torch.eye(test_c2w.shape[-1], device=test_c2w.device) * eps
            test_w2c = torch.linalg.inv(c2w_stable)

        # Ensure viewmat is (4, 4) - squeeze if needed
        viewmat = test_w2c.squeeze(0) if test_w2c.dim() == 3 else test_w2c

        # Extract intrinsics as Python floats
        fx = float(test_intr[0].item())
        fy = float(test_intr[1].item())
        cx = float(test_intr[2].item())
        cy = float(test_intr[3].item())

        # Project 3D gaussians to 2D
        xys, depths, radii, conics, compensation, cov3d = project_gaussians(
            means3d=xyz,
            scales=scale,
            glob_scale=1.0,
            quats=rotation,
            viewmat=viewmat,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            img_height=H,
            img_width=W,
            clip_thresh=0.01,
        )

        # Prepare colors: flatten SH features to (N, (sh_degree+1)^2 * 3)
        if feature.dim() == 3:
            n_gaussians, n_coeffs, n_channels = feature.shape
            colors = feature.reshape(n_gaussians, n_coeffs * n_channels)
        else:
            colors = feature

        # Background color (black)
        n_channels = colors.shape[1] if colors.dim() > 1 else 3
        background = torch.zeros(n_channels, device=xyz.device)
        if n_channels < 3:
            background = F.pad(background, (0, 3 - n_channels))

        # Rasterize using simple loop
        rendered = rasterize_gaussians_simple_loop(
            xys=xys,
            depths=depths,
            radii=radii,
            conics=conics,
            colors=colors,
            opacity=opacity,
            img_height=H,
            img_width=W,
            block_width=16,
            background=background,
        )

        # rendered is (H, W, channels), need (1, H, W, channels)
        rendered = rendered.unsqueeze(0)

        return rendered  # (1, H, W, C)

    @staticmethod
    def forward(
        ctx,
        xyz,
        feature,
        scale,
        rotation,
        opacity,
        test_c2ws,
        test_intr,
        W,
        H,
        sh_degree,
        near_plane,
        far_plane,
    ):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr)
        ctx.W = W
        ctx.H = H
        ctx.sh_degree = sh_degree
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane

        with torch.no_grad():
            B, V, _ = test_intr.shape
            renderings = torch.zeros(B, V, H, W, 4, device=xyz.device)  # RGBA
            for ib in range(B):
                for iv in range(V):
                    renderings[ib, iv : iv + 1] = GaussianRenderer.render(
                        xyz[ib],
                        feature[ib],
                        scale[ib],
                        rotation[ib],
                        opacity[ib],
                        test_c2ws[ib, iv],
                        test_intr[ib, iv],
                        W,
                        H,
                        sh_degree,
                        near_plane,
                        far_plane,
                    )
        renderings = renderings.requires_grad_()
        return renderings

    @staticmethod
    def backward(ctx, grad_output):
        xyz, feature, scale, rotation, opacity, test_c2ws, test_intr = ctx.saved_tensors
        return (
            torch.zeros_like(xyz),
            torch.zeros_like(feature),
            torch.zeros_like(scale),
            torch.zeros_like(rotation),
            torch.zeros_like(opacity),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
