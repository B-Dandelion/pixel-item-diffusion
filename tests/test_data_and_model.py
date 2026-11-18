"""Offline invariant tests; no real dataset training or optimizer steps."""

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.analyze_candidates import inspect
from scripts.audit_images import inspect_images
from scripts.collect_candidates import eligible_urls, extract_zip, save_original, select_candidates
from src.diffusion import DDPM
from src.unet import UNet


class DataTests(unittest.TestCase):
    def test_default_collection_does_not_reintroduce_archived_data(self):
        active = dict(source_id="food", kind="oga", downloads=["pixel.zip", "vector.zip"],
                      deferred_downloads=["vector.zip"], review_status="active_review")
        held = dict(source_id="tiles", kind="kenney", review_status="deferred_scope_or_style")
        hf = dict(source_id="synthetic", kind="hf")
        self.assertEqual(select_candidates([active, held, hf]), [active])
        self.assertEqual(select_candidates([active, held], ids=["tiles"]), [])
        self.assertEqual(select_candidates([active, held], ids=["tiles"], include_deferred=True), [held])
        self.assertEqual(eligible_urls(active), ["pixel.zip"])
        self.assertEqual(eligible_urls(active, include_deferred=True), ["pixel.zip", "vector.zip"])

    def test_raw_audit_corruption_duplicates_and_non_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGBA", (34, 34), (255, 0, 0, 255)).save(root / "W_Sword.png")
            (root / "W_Copy.png").write_bytes((root / "W_Sword.png").read_bytes())
            (root / "broken.png").write_bytes(b"not an image")
            (root / "license.txt").write_text("fixture")
            _, summary = inspect_images(root)
            self.assertEqual((summary["png_files"], summary["unreadable"], summary["exact_duplicates"]), (3, 1, 1))
            self.assertEqual(summary["ignored_non_png"], ["license.txt"])

    def test_visible_dedup_ignores_invisible_rgb_but_preserves_color(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = np.zeros((4, 4, 4), dtype=np.uint8)
            a[1, 1] = [255, 0, 0, 255]
            Image.fromarray(a).save(root / "a.png")
            a[0, 0] = [20, 30, 40, 0]
            Image.fromarray(a).save(root / "b.png")
            a[1, 1] = [0, 255, 0, 255]
            Image.fromarray(a).save(root / "c.png")
            results = [inspect(root / name, "fixture") for name in ["a.png", "b.png", "c.png"]]
            self.assertNotEqual(results[0]["sha256"], results[1]["sha256"])
            self.assertEqual(results[0]["canonical_rgba_sha256"], results[1]["canonical_rgba_sha256"])
            self.assertNotEqual(results[0]["canonical_rgba_sha256"], results[2]["canonical_rgba_sha256"])

    def test_archive_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("../outside.txt", b"unsafe")
            with self.assertRaises(ValueError):
                extract_zip(archive, root / "extracted")
            self.assertFalse((root / "outside.txt").exists())

    def test_original_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.png"
            save_original(path, b"original")
            save_original(path, b"original")
            with self.assertRaises(ValueError):
                save_original(path, b"different")
            self.assertEqual(path.read_bytes(), b"original")

    def test_synthetic_policy_and_identification_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "i-example.png"
            Image.new("RGBA", (32, 32), "red").save(path)
            self.assertEqual(inspect(path, "hf_atlas")["final_inclusion"], "research_only_excluded_by_user_policy")
            self.assertEqual(inspect(path, "hf_krea16", allow_synthetic_review=True)["final_inclusion"],
                             "pending_synthetic_review_not_training_approved")
            self.assertEqual(inspect(path, "dcss_cc0")["role"], "identification_symbol")

    def test_viewer_jpeg_is_not_reported_as_original_png(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.png"
            Image.new("RGB", (24, 24), "blue").save(path, format="JPEG")
            result = inspect(path, "hf_pixelgpt")
            self.assertEqual(result["actual_format"], "JPEG")
            self.assertEqual(result["extension_format_mismatch"], 1)
            self.assertIn("viewer_preview_not_verified_lossless_original", result["quality_flags"])


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_unet_shape_and_timestep_contract(self):
        # Small width tests the resolution/skip contract; full-width GPU probe is separate.
        model = UNet(base_channels=8)
        output = model(torch.zeros(2, 3, 32, 32), torch.tensor([0, 999]))
        self.assertEqual(tuple(output.shape), (2, 3, 32, 32))
        self.assertTrue(torch.isfinite(output).all())
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3, 32, 32), torch.tensor([0]))

    def test_forward_diffusion_coefficients_and_oracle_final_step(self):
        diffusion = DDPM(steps=10)
        clean = torch.linspace(-0.7, 0.7, 12).reshape(1, 3, 2, 2)
        noise = torch.full_like(clean, 0.3)
        t = torch.tensor([0])
        noisy = diffusion.q_sample(clean, t, noise)
        expected = (1 - 1e-4) ** 0.5 * clean + 1e-2 * noise
        torch.testing.assert_close(noisy, expected)
        self.assertEqual(diffusion.posterior_variance[0].item(), 0.0)
        result = diffusion.p_sample(lambda x, step: noise, noisy, t)
        torch.testing.assert_close(result, clean, atol=1e-6, rtol=1e-5)
        self.assertTrue((diffusion.alpha_bars[1:] < diffusion.alpha_bars[:-1]).all())


if __name__ == "__main__":
    unittest.main()
