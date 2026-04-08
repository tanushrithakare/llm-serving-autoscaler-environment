import os
import sys

# Ensure the root directory is on the path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from environment import LLMServeEnv  # type: ignore # noqa: E402
from grader import LLMServeGrader  # type: ignore # noqa: E402
from inference import ReactiveController  # type: ignore # noqa: E402

def grade_all():
    grader = LLMServeGrader()
    
    def make_agent(task):
        controller = ReactiveController(task)
        _last_reward = [0.0]  # closure-safe storage
        
        def agent(obs, reward=None):
            if reward is not None:
                _last_reward[0] = reward
            action = controller.act(obs, _last_reward[0])
            return action
        return agent

    # Special wrapper to handle the reward from the grader's loop
    def run_grade(agent_fn, task_name):
        env = LLMServeEnv()
        obs = env.reset(task_name)
        done = False
        last_r = 0.0
        while not done:
            action = agent_fn(obs, last_r)
            obs, last_r, done, _ = env.step(action)
        return env.episode_stats()

    grader = LLMServeGrader()
    
    e_stats = run_grade(make_agent("easy"), "easy")
    m_stats = run_grade(make_agent("medium"), "medium")
    h_stats = run_grade(make_agent("hard"), "hard")

    e_score = grader._compute_score(e_stats)
    m_score = grader._compute_score(m_stats)
    h_score = grader._compute_score(h_stats)

    print("=" * 40)
    print("FINAL OPTIMALITY CHECK: GRADER DIRECT")
    print("=" * 40)
    print(f"Easy   : {e_score:.4f}")
    print(f"Medium : {m_score:.4f}")
    print(f"Hard   : {h_score:.4f}")
    print("-" * 40)
    print(f"Overall: {(e_score + m_score + h_score) / 3:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    grade_all()
