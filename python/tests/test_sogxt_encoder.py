"""Unit tests for the SOG-XT (KISS-GS) container encoder."""

import json

import numpy as np

from sogs.sogxt_encoder import morton_order_sort, run_sogxt_compression


def _make_splats(n=500, seed=123):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(n, 3)).astype(np.float32) * 2.0
    opacities = rng.normal(size=(n,)).astype(np.float32)  # logit
    scales = (rng.normal(size=(n, 3)) * 0.5).astype(np.float32)  # log
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    f_dc = (rng.normal(size=(n, 3)) * 0.5).astype(np.float32)
    f_rest = (rng.normal(size=(n, 45)) * 0.2).astype(np.float32)
    return {
        "means": means,
        "opacities": opacities,
        "scales": scales,
        "quats": quats,
        "f_dc": f_dc,
        "f_rest": f_rest,
    }


class TestSogXtEncoder:
    def test_container_files_and_meta(self, tmp_path):
        """Encoding writes meta.json + the expected WebP planes."""
        run_sogxt_compression(tmp_path, _make_splats(), iterations=3, sh_codebook_side=16)

        expected_files = {
            "meta.json",
            "scene.json",
            "active_mask.webp",
            "means_bytes_0.webp",
            "means_bytes_1.webp",
            "opacities.webp",
            "scales.webp",
            "quaternions.webp",
            "f_dc.webp",
            "f_rest_centroids.webp",
            "f_rest_labels.webp",
        }
        actual = {p.name for p in tmp_path.iterdir()}
        assert expected_files <= actual

        meta = json.loads((tmp_path / "meta.json").read_text())
        assert meta["format"] == "sog-xt"
        assert meta["version"] == 3
        assert meta["count"] == 500
        assert meta["gridSide"] ** 2 >= 500
        assert meta["shN"]["layout"] == "uv-codebook"
        assert len(meta["shN"]["centroidsMins"]) == 45
        assert len(meta["shN"]["centroidsMaxs"]) == 45

    def test_grid_side_matches_count(self, tmp_path):
        splats = _make_splats(n=1000)
        run_sogxt_compression(tmp_path, splats, iterations=2, sh_codebook_side=16)
        meta = json.loads((tmp_path / "meta.json").read_text())
        gs = meta["gridSide"]
        assert gs * gs >= 1000
        assert gs % 4 == 0

    def test_reference_decoder_roundtrip(self, tmp_path):
        """Container decodes back to ~the input attributes (allowing quantization)."""
        splats = _make_splats(n=800, seed=99)
        run_sogxt_compression(tmp_path, splats, iterations=4, sh_codebook_side=16)

        meta = json.loads((tmp_path / "meta.json").read_text())
        gs = meta["gridSide"]
        cs = meta["shN"]["centroidSide"]

        from PIL import Image

        def read_u8(name):
            img = np.array(Image.open(tmp_path / f"{name}.webp"))
            if img.ndim == 2:
                img = img[..., None]
            return img.astype(np.uint8)

        def dequantize(q, mn, mx):
            q = q.astype(np.float32)
            return (q - q.min()) / (q.max() - q.min() + 1e-8) * (mx - mn) + mn

        mask = read_u8("active_mask")[..., 0].reshape(-1) > 0

        # means
        lo = read_u8("means_bytes_0").astype(np.float32)
        hi = read_u8("means_bytes_1").astype(np.float32)
        msl = dequantize(lo + 256.0 * hi, meta["means"]["mins"], meta["means"]["maxs"])
        means = (np.sign(msl) * np.expm1(np.abs(msl))).reshape(gs * gs, 3)[mask]

        # opacities (logit)
        op = dequantize(
            read_u8("opacities")[..., 0], meta["opacities"]["mins"], meta["opacities"]["maxs"]
        )
        op = op.reshape(-1)[mask]

        # scales (log)
        smn = np.asarray(meta["scales"]["mins"]).reshape(1, 1, 3)
        smx = np.asarray(meta["scales"]["maxs"]).reshape(1, 1, 3)
        scales = np.log(np.exp(dequantize(read_u8("scales"), smn, smx)).reshape(gs * gs, 3)[mask])

        # quats
        qmn = np.asarray(meta["quats"]["mins"]).reshape(1, 1, 4)
        qmx = np.asarray(meta["quats"]["maxs"]).reshape(1, 1, 4)
        quats = dequantize(read_u8("quaternions"), qmn, qmx).reshape(gs * gs, 4)[mask]

        # f_dc
        dmn = np.asarray(meta["sh0"]["mins"]).reshape(1, 1, 3)
        dmx = np.asarray(meta["sh0"]["maxs"]).reshape(1, 1, 3)
        f_dc = dequantize(read_u8("f_dc"), dmn, dmx).reshape(gs * gs, 3)[mask]

        # f_rest via codebook
        lab = read_u8("f_rest_labels").astype(np.int64)
        ci = (lab[..., 1] * cs + lab[..., 0]).reshape(-1)
        cmn = np.asarray(meta["shN"]["centroidsMins"], np.float32)
        cmx = np.asarray(meta["shN"]["centroidsMaxs"], np.float32)
        tiled = read_u8("f_rest_centroids").astype(np.float32)
        th, tw, ch = tiled.shape
        h, w = th // 3, tw // 5
        tiles = tiled.reshape(3, h, 5, w, ch).transpose(1, 3, 4, 0, 2).reshape(h, w, ch * 3 * 5)
        tiles = (tiles - tiles.min()) / (tiles.max() - tiles.min() + 1e-8)
        centroids = (
            tiles.reshape(cs, cs, 45) * (cmx - cmn).reshape(1, 1, 45) + cmn.reshape(1, 1, 45)
        ).reshape(cs * cs, 45)
        f_rest = centroids[ci].reshape(gs * gs, 45)[mask]

        # Apply the same morton sort to the originals for comparison
        orig = _make_splats(n=800, seed=99)
        idx = morton_order_sort(orig["means"])
        means_o = orig["means"][idx]
        op_o = orig["opacities"][idx]
        scales_o = orig["scales"][idx]
        quats_o = orig["quats"][idx]
        f_dc_o = orig["f_dc"][idx]
        f_rest_o = orig["f_rest"][idx]

        def assert_close(name, dec, ref, tol):
            err = np.abs(dec - ref).max()
            assert err < tol, f"{name} max err {err} >= {tol}"

        assert_close("means", means, means_o, 0.01)
        assert_close("opacities", op, op_o, 0.05)
        assert_close("scales", scales, scales_o, 0.05)
        assert_close("quats", quats, quats_o, 0.02)
        assert_close("f_dc", f_dc, f_dc_o, 0.05)
        # f_rest is vector-quantized, so allow a much larger tolerance
        assert_close("f_rest", f_rest, f_rest_o, 1.5)
