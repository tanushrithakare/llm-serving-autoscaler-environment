import pytest
import numpy as np
from src.environment import LLMServeEnv
from src.models import LLMServeAction, LLMServeObs
from src.grader import LLMServeGrader

def test_reset():
    """Verify reset returns a valid LLMServeObs."""
    env = LLMServeEnv()
    obs = env.reset(task="easy")
    assert isinstance(obs, LLMServeObs)
    assert obs.active_gpus == 4
    assert obs.queue_length == 0

def test_determinism():
    """Verify environment is deterministic given the same seed (42)."""
    env1 = LLMServeEnv()
    env2 = LLMServeEnv()
    
    obs1 = env1.reset(task="medium")
    obs2 = env2.reset(task="medium")
    
    # Check initial state
    assert obs1 == obs2
    
    # Check next 10 steps
    for _ in range(10):
        action = LLMServeAction(scale=0, batch_size=64, spot_allocation=0.0)
        o1, r1, d1, i1 = env1.step(action)
        o2, r2, d2, i2 = env2.step(action)
        assert o1 == o2
        assert r1 == r2
        assert d1 == d2

def test_full_episode():
    """Verify a full 1000-step episode completes."""
    env = LLMServeEnv()
    env.reset(task="hard")
    
    done = False
    count = 0
    while not done:
        action = LLMServeAction(scale=1, batch_size=128, spot_allocation=0.5)
        obs, reward, done, info = env.step(action)
        count += 1
        assert -1.0 <= reward <= 1.0
        
    assert count == 1000
    assert done is True

def test_grader():
    """Verify grader returns a score in [0.0, 1.0]."""
    grader = LLMServeGrader()
    
    # Simple lambda agent
    def dummy_agent(obs):
        return LLMServeAction(scale=0, batch_size=64, spot_allocation=0.1)
    
    score = grader.grade(dummy_agent, task="easy")
    assert 0.0 <= score <= 1.0

def test_spot_preemption_logged():
    """Verify spot preemption events are recorded in info dictionary (task=medium)."""
    env = LLMServeEnv()
    # We need to run enough steps to likely trigger a 3% event
    env.reset(task="medium")
    
    found_preemption = False
    for _ in range(200):
        # Action with high spot allocation to make it eligible
        action = LLMServeAction(scale=0, batch_size=64, spot_allocation=0.9)
        _, _, _, info = env.step(action)
        if info.get("preempted_gpus", 0) > 0:
            found_preemption = True
            break
    
    # Since seed is 42, deterministic check:
    # First preemption on task=medium step 12 for seed 42
    assert found_preemption is True
