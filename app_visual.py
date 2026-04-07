import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import time
import os
import sys

# Ensure the root directory is on the path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from environment import LLMServeEnv  # type: ignore # noqa: E402
from baseline import PPOAgent  # type: ignore # noqa: E402

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Cluster Visualizer",
    page_icon="[Cluster]",
    layout="wide",
)

# Deep dark styling additions
st.markdown("""
<style>
    .agent-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 25px; font-size: 18px; margin-bottom: 40px;}
    .decision-container { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 30px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }
    .decision-box { text-align: center; padding: 10px; }
    .decision-label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px; }
    .decision-value { font-size: 1.8rem; font-weight: 700; color: #e2e8f0; }
    .scale-up { color: #22c55e !important; }
    .scale-down { color: #ef4444 !important; }
    .scale-hold { color: #facc15 !important; }
    .header-spacing { margin-top: 40px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Simulation State - ROBUST INITIALIZATION
# -----------------------------------------------------------------------------
if 'initialized' not in st.session_state:
    st.session_state.env = LLMServeEnv()
    st.session_state.agent = PPOAgent()
    st.session_state.obs = st.session_state.env.reset(task="medium")
    
    st.session_state.is_running = False
    st.session_state.step = 0
    st.session_state.total_reward = 0.0
    st.session_state.last_reward = 0.0
    st.session_state.last_action = None
    st.session_state.history = [{
        'Step': 0,
        'Active GPUs': st.session_state.obs.active_gpus,
        'Queue Length': st.session_state.obs.queue_length,
        'Latency (ms)': st.session_state.obs.avg_latency,
        'Total Reward': 0.0,
    }]
    st.session_state.initialized = True

def reset_simulation(task: str):
    st.session_state.obs = st.session_state.env.reset(task=task)
    st.session_state.is_running = False
    st.session_state.step = 0
    st.session_state.total_reward = 0.0
    st.session_state.last_reward = 0.0
    st.session_state.last_action = None
    st.session_state.history = [{
        'Step': 0,
        'Active GPUs': st.session_state.obs.active_gpus,
        'Queue Length': st.session_state.obs.queue_length,
        'Latency (ms)': st.session_state.obs.avg_latency,
        'Total Reward': 0.0,
    }]

def step_simulation():
    action = st.session_state.agent(st.session_state.obs)
    st.session_state.last_action = action
    
    next_obs, reward, done, _ = st.session_state.env.step(action)
    
    st.session_state.step += 1
    st.session_state.last_reward = reward
    st.session_state.total_reward += reward
    st.session_state.obs = next_obs
    
    st.session_state.history.append({
        'Step': st.session_state.step,
        'Active GPUs': next_obs.active_gpus,
        'Queue Length': next_obs.queue_length,
        'Latency (ms)': next_obs.avg_latency,
        'Total Reward': st.session_state.total_reward
    })
    
    if done:
        st.session_state.is_running = False


# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Overlord Mainframe")
    
    task_mode = st.selectbox("1. TRAFFIC PROFILE", ["easy", "medium", "hard"], index=1)
    if st.button("RESTART CLUSTER", use_container_width=True):
        reset_simulation(task_mode)
        st.rerun()
        
    st.divider()
    
    st.markdown("### 2. SIMULATION PLAYBACK")
    if st.session_state.is_running:
        if st.button("PAUSE", use_container_width=True):
            st.session_state.is_running = False
            st.rerun()
    else:
        if st.button("START AUTO-PLAY", type="primary", use_container_width=True):
            st.session_state.is_running = True
            st.rerun()
            
    if st.button("STEP FORWARD", disabled=st.session_state.is_running, use_container_width=True):
        step_simulation()
        
    sim_speed = st.slider("SPEED (FPS)", min_value=1, max_value=20, value=6)

# -----------------------------------------------------------------------------
# Uncrowded, Full-Width Main Dashboard Layout
# -----------------------------------------------------------------------------
obs = st.session_state.obs
act = st.session_state.last_action

st.title("LLM Request Routing & Autoscaling")
st.markdown("---")

# ==========================================
# SECTION 1: AGENT DECISION (3-box horizontal layout)
# ==========================================
st.markdown("<h3 class='header-spacing'>🧠 Agent Decision <span style='font-size: 0.8rem; font-weight: normal; color: #94a3b8;'>(Last Action Taken)</span></h3>", unsafe_allow_html=True)

if act:
    scale_class = "scale-hold"
    scale_text = "0"
    if act.scale > 0:
        scale_class = "scale-up"
        scale_text = "+1"
    elif act.scale < 0:
        scale_class = "scale-down"
        scale_text = "-1"
        
    st.markdown(f"""
    <div class="decision-container">
        <div style="display: flex; justify-content: space-around; align-items: center;">
            <div class="decision-box">
                <div class="decision-label">Scale Action</div>
                <div class="decision-value {scale_class}">{scale_text}</div>
            </div>
            <div style="border-left: 1px solid #334155; height: 40px;"></div>
            <div class="decision-box">
                <div class="decision-label">Batch Size</div>
                <div class="decision-value">{act.batch_size}</div>
            </div>
            <div style="border-left: 1px solid #334155; height: 40px;"></div>
            <div class="decision-box">
                <div class="decision-label">Spot Allocation</div>
                <div class="decision-value">{int(act.spot_allocation * 100)}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("System initializing... Press Auto-Play to begin monitoring agent logic.")

# ==========================================
# SECTION 2: GLOBAL METRICS (Wide and evenly spaced)
# ==========================================
st.markdown("<h3 class='header-spacing'>📡 Live System State</h3>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Active GPUs", f"{obs.active_gpus}", delta=f"{obs.incoming_rate:.1f} Requests arriving/sec", delta_color="off")
with col2:
    st.metric("Pending Request Queue", f"{obs.queue_length}")
with col3:
    st.metric("Gateway Latency", f"{obs.avg_latency:.0f} ms", delta=f"{st.session_state.last_reward:+.2f} Latest Step Reward")
with col4:
    st.metric("Cumulative Game Score", f"{st.session_state.total_reward:.2f}")

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# SECTION 3: VISUALIZATIONS (Vertical Stacking to prevent cramming)
# ==========================================
st.markdown("<h3 class='header-spacing'>Performance Telemetry</h3>", unsafe_allow_html=True)

df = pd.DataFrame(st.session_state.history) if 'history' in st.session_state else pd.DataFrame()
if not df.empty:
    df.set_index('Step', inplace=True)

    # We stack the charts vertically to keep them wide, readable, and perfectly uncrowded.
    st.markdown("**1. Hardware Allocation (GPUs)**")
    st.line_chart(df[['Active GPUs']], height=250, color="#4CAF50")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("**2. Response Latency Health (ms)**")
    st.line_chart(df[['Latency (ms)']], height=250, color="#F44336")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("**3. Value Accrual (Total Score)**")
    st.line_chart(df[['Total Reward']], height=250, color="#2196F3")


# -----------------------------------------------------------------------------
# Auto-Play Loop
# -----------------------------------------------------------------------------
if st.session_state.is_running:
    if st.session_state.step < 1000:
        step_simulation()
        time.sleep(1.0 / sim_speed)
        st.rerun()
    else:
        st.session_state.is_running = False
        st.success("SIMULATION COMPLETED.")
