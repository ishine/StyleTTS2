"""Tests for losses.py"""

import torch

from losses import (
    SpectralConvergengeLoss,
    discriminator_loss,
    discriminator_TPRLS_loss,
    feature_loss,
    generator_loss,
    generator_TPRLS_loss,
)


class TestFeatureLoss:
    def test_zero_loss(self):
        # Same feature maps should give zero loss
        fmap = [[torch.randn(2, 32, 100)]]
        loss = feature_loss(fmap, fmap)
        assert loss.item() == 0.0

    def test_positive_loss(self):
        fmap_r = [[torch.zeros(2, 32, 100)]]
        fmap_g = [[torch.ones(2, 32, 100)]]
        loss = feature_loss(fmap_r, fmap_g)
        assert loss.item() > 0.0

    def test_multiple_feature_maps(self):
        fmap_r = [
            [torch.randn(2, 32, 100), torch.randn(2, 64, 50)],
            [torch.randn(2, 128, 25)],
        ]
        fmap_g = [
            [torch.randn(2, 32, 100), torch.randn(2, 64, 50)],
            [torch.randn(2, 128, 25)],
        ]
        loss = feature_loss(fmap_r, fmap_g)
        assert loss.item() > 0.0


class TestDiscriminatorLoss:
    def test_output_format(self):
        disc_real = [torch.ones(2, 1)]
        disc_gen = [torch.zeros(2, 1)]
        loss, r_losses, g_losses = discriminator_loss(disc_real, disc_gen)

        assert isinstance(loss, torch.Tensor)
        assert isinstance(r_losses, list)
        assert isinstance(g_losses, list)
        assert len(r_losses) == 1
        assert len(g_losses) == 1

    def test_perfect_discrimination(self):
        # Real = 1, Generated = 0: should be low loss
        disc_real = [torch.ones(2, 1)]
        disc_gen = [torch.zeros(2, 1)]
        loss, _, _ = discriminator_loss(disc_real, disc_gen)
        assert loss.item() == 0.0

    def test_multiple_discriminators(self):
        disc_real = [torch.randn(2, 1), torch.randn(2, 1)]
        disc_gen = [torch.randn(2, 1), torch.randn(2, 1)]
        loss, r_losses, g_losses = discriminator_loss(disc_real, disc_gen)

        assert len(r_losses) == 2
        assert len(g_losses) == 2


class TestGeneratorLoss:
    def test_output_format(self):
        disc_outputs = [torch.randn(2, 1)]
        loss, gen_losses = generator_loss(disc_outputs)

        assert isinstance(loss, torch.Tensor)
        assert isinstance(gen_losses, list)
        assert len(gen_losses) == 1

    def test_perfect_generation(self):
        # Generator outputs 1: discriminator thinks it's real
        disc_outputs = [torch.ones(2, 1)]
        loss, _ = generator_loss(disc_outputs)
        assert loss.item() == 0.0

    def test_multiple_discriminators(self):
        disc_outputs = [torch.randn(2, 1), torch.randn(2, 1)]
        loss, gen_losses = generator_loss(disc_outputs)
        assert len(gen_losses) == 2


class TestSpectralConvergenceLoss:
    def test_zero_loss(self):
        loss_fn = SpectralConvergengeLoss()
        x = torch.randn(2, 100, 80)
        loss = loss_fn(x, x)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_positive_loss(self):
        loss_fn = SpectralConvergengeLoss()
        x = torch.zeros(2, 100, 80)
        y = torch.ones(2, 100, 80)
        loss = loss_fn(x, y)
        assert loss.item() > 0.0

    def test_normalized(self):
        # Loss should be normalized by the magnitude of y
        loss_fn = SpectralConvergengeLoss()
        x = torch.zeros(2, 100, 80)
        y = torch.ones(2, 100, 80)
        loss = loss_fn(x, y)
        # ||y - x|| / ||y|| = ||ones|| / ||ones|| = 1.0
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)


class TestDiscriminatorTPRLSLoss:
    def test_output_type(self):
        disc_real = [torch.randn(2, 100)]
        disc_gen = [torch.randn(2, 100)]
        loss = discriminator_TPRLS_loss(disc_real, disc_gen)
        assert isinstance(loss, torch.Tensor)

    def test_multiple_discriminators(self):
        disc_real = [torch.randn(2, 100), torch.randn(2, 50)]
        disc_gen = [torch.randn(2, 100), torch.randn(2, 50)]
        loss = discriminator_TPRLS_loss(disc_real, disc_gen)
        assert isinstance(loss, torch.Tensor)

    def test_bounded_by_tau(self):
        # TPRLS loss should be bounded by tau (0.04)
        disc_real = [torch.randn(2, 100)]
        disc_gen = [torch.randn(2, 100)]
        loss = discriminator_TPRLS_loss(disc_real, disc_gen)
        # Loss = tau - ReLU(tau - L_rel), so loss <= tau
        assert loss.item() <= 0.04 + 1e-6


class TestGeneratorTPRLSLoss:
    def test_output_type(self):
        disc_real = [torch.randn(2, 100)]
        disc_gen = [torch.randn(2, 100)]
        loss = generator_TPRLS_loss(disc_real, disc_gen)
        assert isinstance(loss, torch.Tensor)

    def test_multiple_discriminators(self):
        disc_real = [torch.randn(2, 100), torch.randn(2, 50)]
        disc_gen = [torch.randn(2, 100), torch.randn(2, 50)]
        loss = generator_TPRLS_loss(disc_real, disc_gen)
        assert isinstance(loss, torch.Tensor)

    def test_bounded_by_tau(self):
        # TPRLS loss should be bounded by tau (0.04)
        disc_real = [torch.randn(2, 100)]
        disc_gen = [torch.randn(2, 100)]
        loss = generator_TPRLS_loss(disc_real, disc_gen)
        # Loss = tau - ReLU(tau - L_rel), so loss <= tau
        assert loss.item() <= 0.04 + 1e-6
