import gradio as gr
import httpx
import os

CSS = """
.gradio-container {
    font-family: 'Segoe UI', 'Inter', system-ui, sans-serif !important;
}
.hud-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    color: #e0e0e0;
}
.hud-card-success {
    background: linear-gradient(135deg, #0d2818 0%, #1a4731 100%);
    border: 1px solid #2d6a4f;
    color: #95d5b2;
}
.hud-card-danger {
    background: linear-gradient(135deg, #2d0a0a 0%, #4a1212 100%);
    border: 1px solid #9b2226;
    color: #f4a0a0;
}
.hud-card-warn {
    background: linear-gradient(135deg, #2d1f00 0%, #4a3500 100%);
    border: 1px solid #e09f3e;
    color: #ffd166;
}
.phase-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 600;
}
"""

def create_gradio_ui(server_url: str = "http://localhost:7860"):

    # --- Helper functions (Part 2) ---
    def build_timeline(history):
        """Build formatted investigation timeline from history."""
        if not history:
            return "No investigation steps recorded yet."
        lines = []
        for h in history:
            step = h.get("step", "?")
            tool = h.get("tool", "unknown")
            params = h.get("params", "")
            reward = h.get("reward", 0)
            feedback = h.get("feedback", "")
            status = h.get("status", "INFO")
            
            icon = {"SUCCESS": "🟢", "REJECTED": "🔴", "INFO": "🟡"}.get(status, "⚪")
            sign = "+" if reward > 0 else ""
            lines.append(
                f"**Step {step}** {icon} `{tool}` → `{params}`\n"
                f"  Result: {feedback} ({sign}{reward:.2f})\n"
            )
        return "\n".join(lines)

    def build_reasoning(history):
        """Build agent reasoning visibility panel."""
        if not history:
            return "No reasoning recorded yet. Execute a step to see agent thinking."
        lines = []
        for h in history:
            step = h.get("step", "?")
            tool = h.get("tool", "")
            reasoning = h.get("reasoning", "No reasoning provided")
            reward = h.get("reward", 0)
            icon = "🟢" if reward > 0 else "🔴" if reward < 0 else "🟡"
            lines.append(
                f"### Step {step} — `{tool}` {icon}\n"
                f"**Thought:** {reasoning}\n"
            )
        return "\n---\n".join(lines)

    def build_explanation(history, state_data):
        """Generate a plain-English explanation of the incident."""
        if not history:
            return "Start the investigation to generate an explanation."
        
        status = state_data.get("status", "Active") if state_data else "Active"
        steps = len(history)
        successes = sum(1 for h in history if h.get("reward", 0) > 0)
        
        attack_type = "unknown threat"
        root_file = "not yet identified"
        ioc_val = "not yet confirmed"
        
        for h in history:
            if h.get("tool") == "inspect_file" and h.get("reward", 0) > 0:
                root_file = h.get("params", "unknown")
            if h.get("tool") == "extract_ioc" and h.get("reward", 0) > 0:
                ioc_val = h.get("params", "unknown")
        
        if "sk_live" in ioc_val or "sk_test" in ioc_val:
            attack_type = "production credential exposure"
        elif any(c in ioc_val for c in [".", ":"]):
            if any(d in ioc_val for d in [".cc", ".ru", ".xyz", ".tk", ".onion"]):
                attack_type = "supply-chain backdoor with C2 callback"
            else:
                attack_type = "SQL injection from external IP"
        
        if status == "Mitigated":
            return (
                f"### ✅ Incident Resolved\n\n"
                f"The AI analyst identified a **{attack_type}** originating from `{root_file}`. "
                f"The malicious indicator `{ioc_val}` was confirmed and the threat was neutralized. "
                f"The investigation took **{steps} steps** with **{successes} successful actions**.\n\n"
                f"*In simple terms: A security vulnerability was found and fixed before it could cause damage.*"
            )
        else:
            return (
                f"### 🔍 Investigation In Progress\n\n"
                f"The analyst is investigating a potential **{attack_type}**. "
                f"So far, **{steps} steps** have been taken with **{successes} successful findings**.\n\n"
                f"*In simple terms: The system detected suspicious activity and is working to identify and stop it.*"
            )

    def build_summary(history, state_data):
        """Extract investigation summary from history."""
        if not history:
            return "Investigation has not started."
        
        tools_used = [h["tool"] for h in history]
        successful = [h for h in history if h.get("reward", 0) > 0]
        
        root_cause = "Not identified"
        ioc = "Not confirmed"
        action_taken = "None"
        
        for h in history:
            if h.get("tool") == "inspect_file" and h.get("reward", 0) > 0:
                root_cause = h.get("params", "Unknown")
            if h.get("tool") == "extract_ioc" and h.get("reward", 0) > 0:
                ioc = h.get("params", "Unknown")
            if h.get("tool") == "apply_fix" and h.get("reward", 0) > 0:
                action_taken = "Remediation applied"
        
        confidence = sum(h.get("confidence", 0) for h in history) / max(len(history), 1)
        status = state_data.get("status", "Active") if state_data else "Active"
        
        return (
            f"### Investigation Summary\n"
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| **Status** | {status} |\n"
            f"| **Root Cause** | `{root_cause}` |\n"
            f"| **IOC** | `{ioc}` |\n"
            f"| **Action Taken** | {action_taken} |\n"
            f"| **Confidence** | {confidence:.2f} |\n"
            f"| **Steps Used** | {len(history)} |"
        )

    def build_evaluation(score, history, max_steps):
        """Build kill chain evaluation breakdown."""
        phases = {
            "Reconnaissance": {"done": False, "reward": 0},
            "Identification": {"done": False, "reward": 0},
            "Containment": {"done": False, "reward": 0},
            "Remediation": {"done": False, "reward": 0},
        }
        mistakes = []
        
        for h in history:
            tool = h.get("tool", "")
            reward = h.get("reward", 0)
            
            if tool == "query_logs" and reward > 0:
                phases["Reconnaissance"] = {"done": True, "reward": reward}
            elif tool == "extract_ioc" and reward > 0:
                phases["Identification"] = {"done": True, "reward": reward}
            elif tool == "inspect_file" and reward > 0:
                phases["Containment"] = {"done": True, "reward": reward}
            elif tool == "apply_fix" and reward > 0:
                phases["Remediation"] = {"done": True, "reward": reward}
            
            if reward < 0:
                mistakes.append(f"Step {h.get('step')}: {h.get('feedback', 'Error')} ({reward:+.2f})")
        
        lines = ["### Kill Chain Evaluation\n"]
        for phase, data in phases.items():
            icon = "✅" if data["done"] else "⬜"
            reward_str = f"+{data['reward']:.2f}" if data["done"] else "pending"
            lines.append(f"{icon} **{phase}** — {reward_str}")
        
        efficiency = len(history) / max(max_steps, 1)
        penalty = efficiency * 0.15
        lines.append(f"\n**Efficiency**: {len(history)}/{max_steps} steps ({(1-efficiency)*100:.0f}% budget remaining)")
        lines.append(f"**Efficiency Penalty**: -{penalty:.3f}")
        
        if score is not None:
            lines.append(f"\n### 🏆 **Final Score: {score:.3f}**")
        
        if mistakes:
            lines.append("\n### ⚠️ Analyst Errors")
            for m in mistakes:
                lines.append(f"- {m}")
        else:
            lines.append("\n✅ **No errors recorded — clean investigation**")
        
        return "\n".join(lines)

    def build_severity(state_data):
        """Determine incident severity from state."""
        if not state_data:
            return "UNKNOWN"
        status = state_data.get("status", "Active")
        remaining = state_data.get("steps_remaining", 0)
        
        if status == "Mitigated":
            return "🟢 RESOLVED"
        elif remaining <= 3:
            return "🔴 CRITICAL — budget nearly exhausted"
        elif remaining <= 8:
            return "🟡 HIGH — investigation in progress"
        else:
            return "🟠 MEDIUM — initial assessment"

    # --- Build UI ---
    with gr.Blocks(title="Sentinel-SOC: AI Security Analyst", css=CSS, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
        gr.Markdown("# 🛡️ SENTINEL-SOC: AI Security Analyst System")
        gr.Markdown(
            "> **⚠️ LIVE INCIDENT: Potential Security Breach Detected** — "
            "System is in active investigation mode. AI Analyst is ready for tasking."
        )
        gr.Markdown("🔍 *AI Analyst autonomously investigating using structured forensic tools and kill chain reasoning*")
        
        with gr.Row():
            # LEFT PANEL: Controls + Status
            with gr.Column(scale=1):
                gr.Markdown("### 📊 INCIDENT STATUS")
                status_box = gr.Markdown("**Status**: Active", elem_classes=["hud-card"])
                severity_box = gr.Markdown("**Severity**: MEDIUM", elem_classes=["hud-card-warn"])
                reward_box = gr.Markdown("**Efficiency Score**: 0.00", elem_classes=["hud-card"])
                steps_box = gr.Markdown("**Steps Remaining**: 20", elem_classes=["hud-card"])
                
                gr.Markdown("### ⚙️ SCENARIO")
                task_dropdown = gr.Radio(
                    choices=["easy", "medium", "hard"], 
                    value="easy", 
                    label="Difficulty Tier"
                )
                reset_btn = gr.Button("🔄 INITIALIZE ENVIRONMENT", variant="primary")

                gr.Markdown("---")
                gr.Markdown("### 🔧 FORENSIC TOOLS")
                tool_dropdown = gr.Dropdown(
                    choices=["query_logs", "extract_ioc", "inspect_file", "apply_fix"],
                    value="query_logs",
                    label="Select Tool"
                )
                params_input = gr.Textbox(
                    label="Parameters",
                    placeholder="e.g. access.log, 192.168.1.137, vendor/auth_lib.py",
                    lines=1
                )
                reasoning_input = gr.Textbox(
                    label="Analyst Reasoning",
                    placeholder="Explain your investigative logic...",
                    lines=2
                )
                step_btn = gr.Button("▶️ EXECUTE STEP", variant="secondary")
                gr.Markdown("---")
                grade_btn = gr.Button("📋 FINALIZE & GRADE", variant="stop")
            
            # RIGHT PANEL: Forensic Data + Analysis
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("📄 System Logs"):
                        logs_output = gr.Code(label="Raw Telemetry", language=None, lines=15, interactive=False)
                    with gr.TabItem("💻 Source Inspector"):
                        code_output = gr.Code(label="Source Code", language="python", lines=15, interactive=False)
                    with gr.TabItem("🕵️ Incident Intel"):
                        thread_output = gr.Markdown(label="Threat Assessment")
                    with gr.TabItem("📜 Timeline"):
                        timeline_output = gr.Markdown("No steps recorded yet.")
                    with gr.TabItem("🧠 Agent Reasoning"):
                        reasoning_output = gr.Markdown("Execute a step to see agent thinking.")
                    with gr.TabItem("📊 Summary"):
                        summary_output = gr.Markdown("Investigation has not started.")
                    with gr.TabItem("🏆 Evaluation"):
                        eval_output = gr.Markdown("Complete the investigation to see evaluation.")
                    with gr.TabItem("💡 Explain Simply"):
                        explain_output = gr.Markdown("Start the investigation to generate an explanation.")

        # --- State management ---
        def fetch_full_state():
            """Fetch state + history and build all UI components."""
            try:
                with httpx.Client(timeout=15) as client:
                    resp = client.get(f"{server_url}/state")
                    if resp.status_code != 200:
                        return ["Error fetching state"] * 12
                    
                    data = resp.json()
                    history_resp = client.get(f"{server_url}/history")
                    history_data = history_resp.json().get("history", [])
                    
                    max_steps = data.get("steps_remaining", 0) + len(history_data)
                    
                    return [
                        f"**Status**: {data.get('status', 'Active')}",
                        f"**Severity**: {build_severity(data)}",
                        f"**Efficiency Score**: {data.get('reward_signal', 0.0):.2f}",
                        f"**Steps Remaining**: {data.get('steps_remaining', 0)}",
                        data.get("logs", ""),
                        data.get("code_snippet", ""),
                        data.get("incident_thread", ""),
                        build_timeline(history_data),
                        build_reasoning(history_data),
                        build_summary(history_data, data),
                        build_evaluation(None, history_data, max_steps),
                        build_explanation(history_data, data),
                    ]
            except Exception as e:
                return [f"ERR: {str(e)}"] * 12

        def on_reset(task):
            try:
                with httpx.Client(timeout=15) as client:
                    client.post(f"{server_url}/reset", params={"task": task})
                return fetch_full_state()
            except Exception as e:
                return [f"ERR: {str(e)}"] * 12

        def on_step(tool, params, reasoning):
            try:
                action = {
                    "reasoning": reasoning or f"Executing {tool}",
                    "tool": tool,
                    "parameters": params or ""
                }
                with httpx.Client(timeout=15) as client:
                    client.post(f"{server_url}/step", json=action)
                return fetch_full_state()
            except Exception as e:
                return [f"ERR: {str(e)}"] * 12

        def on_grade():
            try:
                with httpx.Client(timeout=15) as client:
                    score_resp = client.post(f"{server_url}/grade")
                    score = score_resp.json().get("score", 0)
                    
                    history_resp = client.get(f"{server_url}/history")
                    history_data = history_resp.json().get("history", [])
                    
                    state_resp = client.get(f"{server_url}/state")
                    state_data = state_resp.json()
                    max_steps = state_data.get("steps_remaining", 0) + len(history_data)
                    
                    return build_evaluation(score, history_data, max_steps)
            except Exception as e:
                return f"ERR: {str(e)}"

        # --- Wire events ---
        all_outputs = [
            status_box, severity_box, reward_box, steps_box,
            logs_output, code_output, thread_output,
            timeline_output, reasoning_output, summary_output, eval_output, explain_output
        ]
        
        reset_btn.click(on_reset, inputs=[task_dropdown], outputs=all_outputs)
        step_btn.click(on_step, inputs=[tool_dropdown, params_input, reasoning_input], outputs=all_outputs)
        grade_btn.click(on_grade, outputs=[eval_output])
        demo.load(fetch_full_state, outputs=all_outputs)
        
    return demo
