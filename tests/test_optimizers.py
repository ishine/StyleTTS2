"""Tests for optimizers.py"""

import torch
from torch import nn

from optimizers import MultiOptimizer, build_optimizer, define_scheduler


class TestMultiOptimizer:
    def test_initialization(self):
        model1 = nn.Linear(10, 10)
        model2 = nn.Linear(10, 10)
        opt1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.001)

        multi_opt = MultiOptimizer({"model1": opt1, "model2": opt2})

        assert len(multi_opt.keys) == 2
        assert "model1" in multi_opt.keys
        assert "model2" in multi_opt.keys

    def test_state_dict(self):
        model = nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        multi_opt = MultiOptimizer({"model": opt})

        state_dict = multi_opt.state_dict()

        assert len(state_dict) == 1
        assert state_dict[0][0] == "model"

    def test_zero_grad(self):
        model = nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        multi_opt = MultiOptimizer({"model": opt})

        # Create some gradients
        x = torch.randn(5, 10)
        loss = model(x).sum()
        loss.backward()

        # Verify gradients exist
        assert model.weight.grad is not None

        # Zero gradients
        multi_opt.zero_grad()

        # Verify gradients are zeroed (either None or all zeros)
        assert model.weight.grad is None or torch.all(model.weight.grad == 0)

    def test_zero_grad_specific_key(self):
        model1 = nn.Linear(10, 10)
        model2 = nn.Linear(10, 10)
        opt1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        multi_opt = MultiOptimizer({"model1": opt1, "model2": opt2})

        # Create gradients for both
        x = torch.randn(5, 10)
        loss1 = model1(x).sum()
        loss1.backward()
        loss2 = model2(x).sum()
        loss2.backward()

        # Store model2's gradient sum before zeroing
        model2_grad_sum = model2.weight.grad.sum().item()

        # Zero only model1
        multi_opt.zero_grad(key="model1")

        # model1 should be zeroed, model2 should still have gradients
        assert model1.weight.grad is None or torch.all(model1.weight.grad == 0)
        assert model2.weight.grad is not None
        assert model2.weight.grad.sum().item() == model2_grad_sum

    def test_step(self):
        model = nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        multi_opt = MultiOptimizer({"model": opt})

        # Store initial weights
        initial_weight = model.weight.data.clone()

        # Create gradients and step
        x = torch.randn(5, 10)
        loss = model(x).sum()
        loss.backward()
        multi_opt.step()

        # Weights should have changed
        assert not torch.allclose(model.weight.data, initial_weight)

    def test_param_groups(self):
        model1 = nn.Linear(10, 10)
        model2 = nn.Linear(10, 10)
        opt1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.002)
        multi_opt = MultiOptimizer({"model1": opt1, "model2": opt2})

        # param_groups should contain both optimizers' param_groups
        assert len(multi_opt.param_groups) == 2


class TestDefineScheduler:
    def test_creates_onecyclelr(self):
        model = nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        params = {"max_lr": 0.01, "epochs": 100, "steps_per_epoch": 500}

        scheduler = define_scheduler(opt, params)

        assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

    def test_default_params(self):
        model = nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        # Should work with empty params (using defaults)
        scheduler = define_scheduler(opt, {})

        assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)


class TestBuildOptimizer:
    def test_builds_multi_optimizer(self):
        model1 = nn.Linear(10, 10)
        model2 = nn.Linear(10, 10)
        parameters_dict = {"model1": model1.parameters(), "model2": model2.parameters()}
        scheduler_params = {
            "model1": {"max_lr": 0.01, "epochs": 100, "steps_per_epoch": 500},
            "model2": {"max_lr": 0.02, "epochs": 100, "steps_per_epoch": 500},
        }

        multi_opt = build_optimizer(parameters_dict, scheduler_params, lr=0.001)

        assert isinstance(multi_opt, MultiOptimizer)
        assert len(multi_opt.keys) == 2
        assert len(multi_opt.schedulers) == 2
