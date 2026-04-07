"""
dashboard.py — Real-time visualization for the LLM Serving Autoscaler.

This script runs the baseline agent and uses the environment's render() 
method to display a live ASCII dashboard of the simulation.
"""

import time
import os
import sys

# Ensure the root directory is on the path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from environment import LLMServeEnv  # type: ignore # noqa: E402
from baseline import PPOAgent  # type: ignore # noqa: E402

def clear_screen():
    """Clear the terminal screen based on OS."""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # Attempt to use task from command line, default to 'medium'
    task = sys.argv[1] if len(sys.argv) > 1 else "medium"
    if task not in ["easy", "medium", "hard"]:
        print(f"Error: Unknown task '{task}'. Use: easy | medium | hard")
        return

    env = LLMServeEnv()
    agent = PPOAgent()
    
    obs = env.reset(task=task)
    done = False
    total_reward = 0
    step_count = 0
    
    try:
        while not done:
            # 1. Decide action
            action = agent(obs)
            
            # 2. Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # 3. Render every step (or every N steps for speed)
            if step_count % 1 == 0:
                clear_screen()
                print(f"--- LLM Serving Autoscaler Live Monitor ---")
                env.render(reward=reward)
                print(f" Cumulative Reward: {total_reward:+.4f}")
                print(f" Press Ctrl+C to stop simulation")
                
                # Control animation speed (0.05s = ~20 FPS)
                time.sleep(0.05) 
            
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    
    # Final stats
    stats = env.episode_stats()
    print("\n" + "="*54)
    print(f"  EPISODE COMPLETE: {task.upper()}")
    print("="*54)
    print(f"  Mean Latency:      {stats['mean_latency']:.2f} ms")
    print(f"  Mean Service %:    {stats['mean_service_ratio']:.1%}")
    print(f"  Mean GPU Cost:     {stats['mean_cost']:.4f}")
    print(f"  Final Reward:      {total_reward:.4f}")
    print("="*54)

if __name__ == "__main__":
    main()
