"""
Tests for Sentinel-SOC Forensic Environment.
Validates procedural generation, kill chain enforcement, and grading.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import SentinelSOCEnv
from models import IncidentAction


def test_reset_produces_unique_scenarios():
    """Each reset should produce a different scenario (procedural generation)."""
    env = SentinelSOCEnv()
    obs1 = env.reset(task="easy")
    seed1 = env.episode_seed
    ioc1 = env.target_ioc

    obs2 = env.reset(task="easy")
    seed2 = env.episode_seed
    ioc2 = env.target_ioc

    assert seed1 != seed2, "Episode seeds must differ between resets"
    assert ioc1 != ioc2, "IOCs must differ between episodes (procedural gen)"
    print("[OK] test_reset_produces_unique_scenarios PASSED")


def test_kill_chain_ordering():
    """Agent must follow Recon -> Identify -> Contain -> Remediate."""
    env = SentinelSOCEnv()
    env.reset(task="easy")

    # Trying to apply_fix before any investigation should be rejected
    obs, reward, done, info = env.step(IncidentAction(
        reasoning="Skip everything", tool="apply_fix", parameters="fix"
    ))
    assert reward < 0, "Premature fix should be penalized"
    assert not done, "Should not be done after premature fix"
    print("[OK] test_kill_chain_ordering PASSED")


def test_easy_full_solve():
    """Complete the easy task optimally and verify grading."""
    env = SentinelSOCEnv()
    env.reset(task="easy")

    # Phase 1: Reconnaissance
    obs, r1, done, info = env.step(IncidentAction(
        reasoning="Scan logs", tool="query_logs", parameters="access.log"
    ))
    assert r1 == 0.1, f"Expected 0.1 for recon, got {r1}"

    # Phase 2: Identification
    obs, r2, done, info = env.step(IncidentAction(
        reasoning="Extract IOC", tool="extract_ioc", parameters=env.target_ioc
    ))
    assert r2 == 0.3, f"Expected 0.3 for IOC, got {r2}"

    # Phase 3: Containment
    obs, r3, done, info = env.step(IncidentAction(
        reasoning="Inspect file", tool="inspect_file", parameters=env.target_file
    ))
    assert r3 == 0.2, f"Expected 0.2 for file, got {r3}"

    # Phase 4: Remediation
    obs, r4, done, info = env.step(IncidentAction(
        reasoning="Apply fix", tool="apply_fix", parameters="rotate_key"
    ))
    assert r4 == 0.4, f"Expected 0.4 for fix, got {r4}"
    assert done, "Should be done after successful fix"

    score = env.grade()
    assert 0.7 < score < 1.0, f"Expected high score, got {score}"
    print(f"[OK] test_easy_full_solve PASSED (score: {score})")


def test_decoy_penalty():
    """Agent should be penalized for falling for decoy indicators."""
    env = SentinelSOCEnv()
    env.reset(task="easy")

    # First, do recon
    env.step(IncidentAction(reasoning="Scan", tool="query_logs", parameters="logs"))

    # Try the decoy IOC
    obs, reward, done, info = env.step(IncidentAction(
        reasoning="Extract", tool="extract_ioc", parameters=env.decoy_ioc
    ))
    assert reward == -0.2, f"Expected -0.2 penalty for decoy, got {reward}"
    print("[OK] test_decoy_penalty PASSED")


def test_noise_scaling():
    """Hard tasks should have more log lines (noise) than easy tasks."""
    env = SentinelSOCEnv()

    env.reset(task="easy")
    easy_lines = len(env.logs.split("\n"))

    env.reset(task="hard")
    hard_lines = len(env.logs.split("\n"))

    assert hard_lines >= easy_lines, f"Hard ({hard_lines} lines) should have >= noise than easy ({easy_lines} lines)"
    print(f"[OK] test_noise_scaling PASSED (easy: {easy_lines} lines, hard: {hard_lines} lines)")


def test_grade_boundaries():
    """Grades must be within [0.01, 0.99] for OpenEnv compliance."""
    env = SentinelSOCEnv()

    # Worst case: no actions
    env.reset(task="easy")
    score = env.grade()
    assert 0.01 <= score <= 0.99, f"Score {score} out of bounds"

    # Best case: perfect solve
    env.reset(task="easy")
    env.step(IncidentAction(reasoning="r", tool="query_logs", parameters="p"))
    env.step(IncidentAction(reasoning="r", tool="extract_ioc", parameters=env.target_ioc))
    env.step(IncidentAction(reasoning="r", tool="inspect_file", parameters=env.target_file))
    env.step(IncidentAction(reasoning="r", tool="apply_fix", parameters="fix"))
    score = env.grade()
    assert 0.01 <= score <= 0.99, f"Score {score} out of bounds"
    print(f"[OK] test_grade_boundaries PASSED (min: 0.01, max: {score})")


def test_all_tasks():
    """Smoke test all task difficulties."""
    env = SentinelSOCEnv()
    for task in ["easy", "medium", "hard"]:
        obs = env.reset(task=task)
        assert obs.status == "Active"
        assert obs.steps_remaining > 0
        assert len(obs.logs) > 0
        
        # Run through optimal path
        env.step(IncidentAction(reasoning="r", tool="query_logs", parameters="p"))
        env.step(IncidentAction(reasoning="r", tool="extract_ioc", parameters=env.target_ioc))
        env.step(IncidentAction(reasoning="r", tool="inspect_file", parameters=env.target_file))
        obs, _, done, _ = env.step(IncidentAction(reasoning="r", tool="apply_fix", parameters="fix"))
        assert done, f"Task {task} should be solved"
        score = env.grade()
        assert score > 0.7, f"Optimal solve of {task} should score > 0.7, got {score}"
        print(f"[OK] test_all_tasks [{task}] PASSED (score: {score})")


if __name__ == "__main__":
    test_reset_produces_unique_scenarios()
    test_kill_chain_ordering()
    test_easy_full_solve()
    test_decoy_penalty()
    test_noise_scaling()
    test_grade_boundaries()
    test_all_tasks()
    print("\n[FINISH] ALL TESTS PASSED")
