"""
BlackHole Optimizer - Pytest Unit Tests
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackhole import BlackHole


class TestBlackHole:
    """Test suite for BlackHole optimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization with default parameters."""
        params = [torch.randn(10, requires_grad=True)]
        opt = BlackHole(params)
        assert opt is not None
    
    def test_initialization_custom(self):
        """Test optimizer initialization with custom parameters."""
        params = [torch.randn(10, requires_grad=True)]
        opt = BlackHole(
            params,
            lr=0.01,
            beta1=0.95,
            beta2=0.99,
            weight_decay=0.001,
            G=0.02,
            c=5.0,
            hbar=0.005,
            k_B=0.02,
            Lambda=0.002,
            alpha=0.1,
            spin=0.3,
            extra_dim_strength=0.02
        )
        assert opt is not None
    
    def test_step(self):
        """Test a single optimization step."""
        params = [torch.randn(10, requires_grad=True)]
        opt = BlackHole(params, lr=0.001)
        
        loss = torch.sum(params[0] ** 2)
        loss.backward()
        opt.step()
        
        assert not torch.isnan(params[0]).any()
    
    def test_multiple_steps(self):
        """Test multiple optimization steps."""
        params = [torch.randn(10, requires_grad=True)]
        opt = BlackHole(params, lr=0.001)
        
        for _ in range(10):
            opt.zero_grad()
            loss = torch.sum(params[0] ** 2)
            loss.backward()
            opt.step()
        
        assert not torch.isnan(params[0]).any()
    
    def test_weight_decay(self):
        """Test weight decay functionality."""
        params = [torch.randn(10, requires_grad=True)]
        opt = BlackHole(params, weight_decay=0.01)
        
        loss = torch.sum(params[0] ** 2)
        loss.backward()
        opt.step()
        
        assert not torch.isnan(params[0]).any()
    
    def test_model_training(self):
        """Test training a simple model."""
        model = nn.Linear(10, 1)
        opt = BlackHole(model.parameters(), lr=0.001)
        
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()
        
        assert not torch.isnan(next(model.parameters()).any())
    
    def test_physics_state(self):
        """Test getting physics state."""
        params = [torch.randn(10, requires_grad=True)]
        opt = BlackHole(params)
        
        loss = torch.sum(params[0] ** 2)
        loss.backward()
        opt.step()
        
        state = opt.get_physics_state()
        assert state is not None
    
    def test_reset_physics(self):
        """Test resetting physics state."""
        params = [torch.randn(10, requires_grad=True)]
        opt = BlackHole(params)
        
        loss = torch.sum(params[0] ** 2)
        loss.backward()
        opt.step()
        
        opt.reset_physics()
        # Check that reset worked by verifying state values
        for state in opt.state.values():
            assert state['mass'] == 1.0
    
    def test_gpu_compatibility(self):
        """Test GPU compatibility if CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        params = [torch.randn(10, device='cuda', requires_grad=True)]
        opt = BlackHole(params, lr=0.001)
        
        loss = torch.sum(params[0] ** 2)
        loss.backward()
        opt.step()
        
        assert not torch.isnan(params[0]).any()
    
    def test_parameter_group(self):
        """Test with parameter groups."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 1)
        )
        
        opt = BlackHole([
            {'params': model[0].parameters(), 'lr': 0.01},
            {'params': model[1].parameters(), 'lr': 0.001}
        ])
        
        assert opt is not None
    
    def test_no_nan_after_many_steps(self):
        """Test stability over many steps."""
        params = [torch.randn(10, requires_grad=True)]
        opt = BlackHole(params, lr=0.001)
        
        for _ in range(100):
            opt.zero_grad()
            loss = torch.sum(params[0] ** 2)
            loss.backward()
            opt.step()
            
            assert not torch.isnan(params[0]).any(), "NaN detected during training"
    
    def test_convergence(self):
        """Test that optimizer actually converges."""
        params = [torch.randn(10, requires_grad=True)]
        opt = BlackHole(params, lr=0.01)
        
        losses = []
        for _ in range(200):
            opt.zero_grad()
            loss = torch.sum(params[0] ** 2)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        
        # Loss should decrease significantly
        assert losses[-1] < losses[0] * 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])