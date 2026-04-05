import streamlit as st
import pandas as pd
import time
from environment import LLMServeEnv
from baseline import BaselineAgent

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Cluster Visualizer",
    page_icon="🖥️",
    layout="wide",
)

# Deep dark styling additions
st.markdown("""
<style>
    .agent-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 25px; font-size: 18px; margin-bottom: 40px;}
    .scale-up { color: #4CAF50; font-weight: bold; }
    .scale-down { color: #F44336; font-weight: bold; }
    .scale-hold { color: #9E9E9E; font-weight: bold; }
    .header-spacing { margin-top: 40px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Simulation State - ROBUST INITIALIZATION
# -----------------------------------------------------------------------------
if 'initialized' not in st.session_state:
    st.session_state.env = LLMServeEnv()
    st.session_state.agent = BaselineAgent()
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
    st.title("🖥️ Overlord Mainframe")
    
    task_mode = st.selectbox("1. TRAFFIC PROFILE", ["easy", "medium", "hard"], index=1)
    if st.button("🔄 RESTART CLUSTER", use_container_width=True):
        reset_simulation(task_mode)
        st.rerun()
        
    st.divider()
    
    st.markdown("### 2. SIMULATION PLAYBACK")
    if st.session_state.is_running:
        if st.button("⏸ PAUSE", use_container_width=True):
            st.session_state.is_running = False
            st.rerun()
    else:
        if st.button("▶️ START AUTO-PLAY", type="primary", use_container_width=True):
            st.session_state.is_running = True
            st.rerun()
            
    if st.button("⏭ STEP FORWARD", disabled=st.session_state.is_running, use_container_width=True):
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
# SECTION 1: GLOBAL METRICS (Wide and evenly spaced)
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
# SECTION 2: THE AGENT'S DECISION (Single uncrowded card)
# ==========================================
st.markdown("<h3 class='header-spacing'>🧠 Agent Action Log (Step {})</h3>".format(st.session_state.step), unsafe_allow_html=True)

if act:
    if act.scale > 0:
        scale_text = "<span class='scale-up'>ADDED 1 GPU SERVER (+1)</span>"
    elif act.scale < 0:
        scale_text = "<span class='scale-down'>REMOVED 1 GPU SERVER (-1)</span>"
    else:
        scale_text = "<span class='scale-hold'>MAINTAINED HARDWARE CAPACITY (0)</span>"
        
    st.markdown(f"""
    <div class='agent-card'>
        The intelligent autoscaler decided to: {scale_text} <br><br>
        • <b>Batch Size Tuned:</b> {act.batch_size} sequences/pass <br>
        • <b>Spot Allocation Mix:</b> {int(act.spot_allocation * 100)}% preemptible instances requested
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("System initializing... Press Auto-Play to begin sequences.")


# ==========================================
# SECTION 3: VISUALIZATIONS (Vertical Stacking to prevent cramming)
# ==========================================
st.markdown("<h3 class='header-spacing'>📊 Performance Telemetry</h3>", unsafe_allow_html=True)

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
        st.success("SIMULATION COMPLETED.", icon="✅")
