"""Unit tests for TemporalUNet3D and SimpleNodeTransformer."""
import unittest
import torch

from src.models import SimpleNodeTransformer, TemporalUNet3D


class ModelArchitectureTests(unittest.TestCase):
    def test_temporal_unet3d_forward_shape(self):
        model = TemporalUNet3D(
            in_channels=1,
            out_channels=32,
            layers=(32, 64, 128),
            gradient_checkpointing=False,
        )
        x = torch.randn(1, 2, 1, 8, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (1, 2, 32, 8, 32, 32))

    def test_simple_node_transformer_forward_shape(self):
        model = SimpleNodeTransformer(
            feat_dim=33,
            hidden_dim=64,
            n_heads=4,
            n_blocks=2,
            dropout=0.0,
        )
        feat_t = torch.randn(5, 33)
        feat_t1 = torch.randn(7, 33)
        coords_t = torch.randn(5, 3)
        coords_t1 = torch.randn(7, 3)
        logits = model(feat_t, feat_t1, coords_t, coords_t1)
        self.assertEqual(logits.shape, (5, 7))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
