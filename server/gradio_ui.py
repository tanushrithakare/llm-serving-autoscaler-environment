import gradio as gr
import httpx
import os
import re
import json
import base64
import numpy as np

CSS = """
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif !important;
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding: 30px 10px !important;
    background-color: #0d0d0d !important;
}

body { background-color: #0d0d0d !important; color: #e0e0e0 !important; }

/* SOC-Grade Cards - NEUTRAL GREY */
.soc-card {
    background: #1c1c1c !important;
    border: 1px solid #444 !important;
    border-radius: 8px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.6) !important;
    min-height: 100px;
}

/* SOC Header Bar - Deep Grey */
.soc-header {
    background: #1c1c1c;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 20px 30px;
    margin-bottom: 30px;
}

.status-dot {
    height: 10px;
    width: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}
.dot-green { background-color: #00ff9c; box-shadow: 0 0 10px #00ff9c; }
.dot-active { background-color: #3b82f6; box-shadow: 0 0 10px #3b82f6; }
.dot-shield { background-color: #777; box-shadow: 0 0 5px #777; }

/* HUD Quick Stats - Grey Baseline */
.hud-card {
    background: #1c1c1c !important;
    border: 1px solid #444 !important;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    min-height: 100px;
}
.hud-card p { margin: 0; color: #888; font-size: 0.85em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.hud-card h3 { margin: 10px 0 0 0; font-size: 1.8em; font-weight: 700; color: #eee; }

.hud-green h3 { color: #00ff9c !important; text-shadow: 0 0 10px rgba(0,255,156,0.2); }
.hud-red h3 { color: #ff4d4f !important; text-shadow: 0 0 10px rgba(255,77,79,0.2); }
.hud-yellow h3 { color: #facc15 !important; text-shadow: 0 0 10px rgba(250,204,21,0.2); }
.hud-blue h3 { color: #00f2ff !important; text-shadow: 0 0 10px rgba(0,242,255,0.2); }

/* Tab styling - Grey Base */
.tabs > .tab-nav { border-bottom: 1px solid #444 !important; gap: 20px !important; margin-bottom: 15px !important; }
.tabs > .tab-nav button { color: #888 !important; font-weight: 600 !important; font-size: 0.9em !important; }
.tabs > .tab-nav button.selected { border-bottom: 2px solid #00f2ff !important; color: #00f2ff !important; background: transparent !important; }

/* Buttons - High Contrast Grey & Neon */
.primary-btn {
    background: #252525 !important;
    color: #00f2ff !important;
    font-weight: 700 !important;
    border: 1px solid #00f2ff !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 12px !important;
}
.primary-btn:hover { background: #00f2ff !important; color: #000 !important; box-shadow: 0 0 15px rgba(0,242,255,0.4); }

.stop-btn {
    background: #1c1c1c !important;
    color: #ff4d4f !important;
    border: 1px solid #ff4d4f !important;
}
.stop-btn:hover { background: #ff4d4f !important; color: #fff !important; }

.secondary-btn {
    background: #252525 !important;
    color: #888 !important;
    border: 1px solid #444 !important;
}

/* Code and Input styles */
.code-wrap { border: 1px solid #444 !important; border-radius: 6px !important; padding: 5px !important; }
textarea, input, select {
    background-color: #111 !important;
    border: 1px solid #444 !important;
    color: #f0f0f0 !important;
    margin-bottom: 10px !important;
}
"""

def create_gradio_ui(server_url: str = "http://localhost:7860"):

    # ── Helper functions (Logical Core) ─────────────────────────────────────

    def build_timeline(history):
        if not history:
            return "No investigation steps recorded yet."
        lines = []
        for h in history:
            step     = h.get("step", "?")
            tool     = h.get("tool", "unknown")
            reward   = h.get("reward", 0)
            feedback = h.get("feedback", "")
            icon     = "🟢" if reward > 0 else "🔴" if reward < 0 else "🟡"
            lines.append(
                f"**Step {step} — `{tool}`** {icon}\n\n"
                f"📄 **Result:** {feedback}\n"
                f"---"
            )
        return "\n".join(lines)

    def build_reasoning(history):
        if not history:
            return "No reasoning recorded yet. See 'BRAIN' tab after taking actions."
        lines = []
        for h in history:
            step      = h.get("step", "?")
            tool      = h.get("tool", "")
            reasoning = h.get("reasoning", "No reasoning provided")
            reward    = h.get("reward", 0)
            icon      = "🟢" if reward > 0 else "🔴" if reward < 0 else "🟡"
            lines.append(
                f"### Step {step} — `{tool}` {icon}\n"
                f"**Analyst Mindset:** {reasoning}\n"
            )
        return "\n---\n".join(lines)

    def build_explanation(history, state_data):
        if not history:
            return "Start the investigation to generate an explanation."
        status   = state_data.get("status", "Active") if state_data else "Active"
        steps    = len(history)
        successes = sum(1 for h in history if h.get("reward", 0) > 0)
        attack_type = "unknown threat"
        root_file   = "not yet identified"
        ioc_val     = "not yet confirmed"
        for h in history:
            if h.get("tool") == "inspect_file" and h.get("reward", 0) > 0:
                root_file = h.get("parameters", "unknown")
            if h.get("tool") == "extract_ioc" and h.get("reward", 0) > 0:
                ioc_val = h.get("parameters", "unknown")
        
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
                f"*Summary: A security vulnerability was found and fixed before it could cause damage.*"
            )
        else:
            return (
                f"### 🔍 Investigation In Progress\n\n"
                f"The analyst is investigating a potential **{attack_type}**. "
                f"So far, **{steps} steps** have been taken with **{successes} successful findings**.\n\n"
                f"*Summary: The system detected suspicious activity and is working to identify and stop it.*"
            )

    def build_summary(history, state_data):
        if not history:
            return "### Investigation Inactive\nSelect severity level and click Reset to start the simulation."

        root_cause   = None
        ioc          = None
        action_taken = None

        for h in history:
            if h.get("tool") == "inspect_file" and h.get("reward", 0) > 0:
                root_cause = h.get("parameters", "Unknown")
            if h.get("tool") == "extract_ioc" and h.get("reward", 0) > 0:
                ioc = h.get("parameters", "Unknown")
            if h.get("tool") == "apply_fix" and h.get("reward", 0) > 0:
                action_taken = True

        raw_status = state_data.get("status", "Active") if state_data else "Active"
        max_steps  = (state_data.get("steps_remaining", 0) if state_data else 0) + len(history)

        # Labels as requested
        status_disp = "🟢 Mitigated" if raw_status == "Mitigated" else "🔴 Threat Ongoing"
        
        def fmt_val(val, icon="🟢"):
            return f"{icon} {val}" if val else "⚠ Pending"

        return (
            f"| Metric | Status |\n"
            f"| :--- | :--- |\n"
            f"| **Current Status** | {status_disp} |\n"
            f"| **Root Cause** | {fmt_val(root_cause)} |\n"
            f"| **IOC Found** | {fmt_val(ioc)} |\n"
            f"| **Remediation** | {fmt_val('Mitigated' if action_taken else None)} |\n"
            f"| **Operations** | {len(history)} / {max_steps} steps |\n\n"
            f"*🔍 See Timeline and Reasoning tabs for detailed decision process.*"
        )


    def build_evaluation(score, history, max_steps):
        phases = ["Reconnaissance", "Identification", "Containment", "Remediation"]
        completed = set()
        for h in history:
            if h.get("reward", 0) > 0:
                tool = h.get("tool", "")
                if tool == "query_logs": completed.add("Reconnaissance")
                elif tool == "extract_ioc": completed.add("Identification")
                elif tool == "inspect_file": completed.add("Containment")
                elif tool == "apply_fix": completed.add("Remediation")
        
        lines = ["### SOC Performance Evaluation\n"]
        for p in phases:
            icon = "🟢" if p in completed else "⬜"
            lines.append(f"{icon} **{p}**")
        
        if score is not None:
            lines.append(f"\n### 🏆 **FINAL GRADE: {score:.3f}**")
        return "\n".join(lines)

    def build_phase_label(history):
        if not history: return "STANDBY"
        completed = set()
        for h in history:
            if h.get("reward", 0) > 0:
                t = h.get("tool", "")
                if t == "query_logs": completed.add("RECON")
                elif t == "extract_ioc": completed.add("IDENT")
                elif t == "inspect_file": completed.add("CONTAIN")
                elif t == "apply_fix": completed.add("REMEDY")
        
        if "REMEDY" in completed: return "RESOLVED"
        if "CONTAIN" in completed: return "REMEDIATING"
        if "IDENT" in completed: return "CONTAINING"
        if "RECON" in completed: return "IDENTIFYING"
        return "RECONNAISSANCE"

    # ── Build UI ─────────────────────────────────────────────────────────────
    with gr.Blocks(title="Sentinel-SOC | AI Security Analyst") as demo:
        # Note: theme and css are now handled at launch/mount in Gradio 6

        # ── SOC HEADER BAR ───────────────────────────────────────────────────
        with gr.Row(elem_classes=["soc-header"]):
            with gr.Column(scale=8):
                gr.Markdown("# 🛡️ SENTINEL-SOC")
                gr.Markdown("*AI agent investigates incidents step-by-step using structured tools*")
            with gr.Column(scale=2, min_width=200):
                gr.HTML(
                    '<div style="text-align:right;">'
                    '<p style="font-size:0.8em; margin:0;"><span class="status-dot dot-green"></span> SYSTEM OPERATIONAL</p>'
                    '<p style="font-size:0.8em; margin:4px 0;"><span class="status-dot dot-active"></span> ANALYST ACTIVE</p>'
                    '<p style="font-size:0.8em; margin:0;"><span class="status-dot dot-shield"></span> KILL CHAIN ENABLED</p>'
                    '</div>'
                )

        # ── QUICK STATS HUD ──────────────────────────────────────────────────
        with gr.Row():
            steps_hud    = gr.HTML(scale=1)
            status_hud   = gr.HTML(scale=1)
        with gr.Row():
            phase_hud    = gr.HTML(scale=1)
            score_hud    = gr.HTML(scale=1)

        gr.Markdown("<br>")

        gr.Markdown("---")

        with gr.Row(equal_height=False):

            # ── LEFT: INCIDENT CONTROLLER ────────────────────────────────────
            with gr.Column(scale=4, min_width=350):
                with gr.Group(elem_classes=["soc-card"]):
                    gr.Markdown("### 🎮 Incident Response Toolkit")
                    
                    task_dropdown = gr.Dropdown(
                        choices=["easy", "medium", "hard"],
                        value="easy",
                        label="Simulation Target"
                    )
                    reset_btn = gr.Button("🔄 Initialize Scenario", variant="secondary", size="md", elem_classes=["secondary-btn"])

                    gr.Markdown("---")
                    
                    tool_dropdown = gr.Dropdown(
                        choices=["query_logs", "extract_ioc", "inspect_file", "apply_fix"],
                        value="query_logs",
                        label="Diagnostic Tool"
                    )
                    params_input = gr.Dropdown(
                        choices=["all", "auth.log", "access.log", "error.log", "system.log"],
                        value="all",
                        label="Target Artifact",
                        allow_custom_value=True
                    )
                    
                    reasoning_input = gr.Textbox(
                        label="Investigative Reasoning",
                        placeholder="Explain your deduction...",
                        lines=3
                    )
                    
                    step_btn = gr.Button("▶ Execute Action", variant="primary", elem_classes=["primary-btn"])
                    grade_btn = gr.Button("📊 Final Review", variant="stop", elem_classes=["stop-btn"])

            # ── RIGHT: TELEMETRY & INTEL ─────────────────────────────────────
            with gr.Column(scale=6):
                with gr.Tabs(elem_classes=["tabs"]):
                    with gr.TabItem("📋 STATUS & EVAL"):
                        summary_output = gr.Markdown(elem_classes=["soc-card"])
                        eval_output = gr.Markdown(elem_classes=["soc-card"])

                    with gr.TabItem("🔍 FORENSIC DATA"):
                        logs_output = gr.Code(label="System Telemetry", language="shell", lines=15, interactive=False)
                        code_output = gr.Code(label="Source Investigation", language="python", lines=12, interactive=False)

                    with gr.TabItem("📜 ANALYSIS TRACE"):
                        timeline_output = gr.Markdown(elem_classes=["soc-card"])
                        reasoning_output = gr.Markdown(elem_classes=["soc-card"])

        # ── STATE LOGIC ──────────────────────────────────────────────────────
        def fmt_hud(label, value, status="info"):
            cls_map = {"success": "hud-green", "danger": "hud-red", "warn": "hud-yellow", "info": "hud-blue"}
            cls = cls_map.get(status, "hud-blue")
            return f'<div class="hud-card {cls}"><p>{label}</p><h3>{value}</h3></div>'

        def fetch_full_state():
            headers = {"Cache-Control": "no-cache", "Pragma": "no-cache"}
            try:
                with httpx.Client(timeout=15) as client:
                    resp = client.get(f"{server_url}/state", headers=headers)
                    if resp.status_code != 200: 
                        return [fmt_hud("ERR", "DISCONNECTED", "danger")] * 4 + ["ERR"] * 7
                    data = resp.json()
                    hist_resp = client.get(f"{server_url}/history", headers=headers)
                    history_data = hist_resp.json().get("history", [])
                    max_steps = data.get("steps_remaining", 0) + len(history_data)
                    
                    # HUD Updates
                    curr_status = data.get('status', 'Active')
                    stat_val = "Mitigated" if curr_status == "Mitigated" else "Ongoing"
                    stat_cls = "success" if curr_status == "Mitigated" else "danger"
                    
                    phase_val = build_phase_label(history_data)
                    phase_cls = "success" if phase_val == "RESOLVED" else "info"
                    
                    score_val = data.get('reward_signal', 0.0)
                    score_cls = "success" if score_val > 0.8 else ("warn" if score_val > 0.4 else "info")

                    return [
                        fmt_hud("STEPS USED", f"{len(history_data)} / {max_steps}", "info"),
                        fmt_hud("THREAT STATUS", stat_val.upper(), stat_cls),
                        fmt_hud("INVESTIGATION PHASE", phase_val, phase_cls),
                        fmt_hud("SCORE", f"{score_val:.2f}", score_cls),
                        data.get("logs", ""),
                        data.get("code_snippet", ""),
                        "\n".join([f"**Step {h['step']} — `{h['tool']}`** {'🟢' if h['reward'] > 0 else '🔴'}\n{h['feedback']}\n---" for h in history_data]),
                        "\n".join([f"### Step {h['step']}\n**Analyst Mindset:** {h['reasoning']}\n---" for h in history_data]),
                        build_summary(history_data, data),
                        build_evaluation(None, history_data, max_steps),
                        ""
                    ]
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                return [fmt_hud("ERR", str(e), "danger")] * 4 + ["ERR"] * 7

        def on_reset(task):
            print(f"DEBUG: Reset triggered with task={task}")
            headers = {"Cache-Control": "no-cache"}
            try:
                with httpx.Client(timeout=15) as client:
                    client.post(f"{server_url}/reset", params={"task": task}, headers=headers)
                return fetch_full_state()
            except Exception: return [fmt_hud("ERR", "RESET FAILED", "danger")] * 4 + ["ERR"] * 7

        def on_step(tool, params, reasoning):
            print(f"DEBUG: Action triggered tool={tool} params={params}")
            headers = {"Cache-Control": "no-cache"}
            try:
                action = {"reasoning": reasoning or f"Executing {tool}", "tool": tool, "parameters": params or ""}
                with httpx.Client(timeout=15) as client:
                    client.post(f"{server_url}/step", json=action, headers=headers)
                return fetch_full_state()
            except Exception: return [fmt_hud("ERR", "STEP FAILED", "danger")] * 4 + ["ERR"] * 7

        def on_grade():
            try:
                with httpx.Client(timeout=15) as client:
                    score = client.post(f"{server_url}/grade").json().get("score", 0)
                    history = client.get(f"{server_url}/history").json().get("history", [])
                    max_steps = client.get(f"{server_url}/state").json().get("steps_remaining", 0) + len(history)
                    return build_evaluation(score, history, max_steps)
            except Exception: return "ERR"

        # ── WIRING ───────────────────────────────────────────────────────────
        all_outputs = [
            steps_hud, status_hud, phase_hud, score_hud,
            logs_output, code_output,
            timeline_output, reasoning_output, summary_output,
            eval_output, gr.Markdown(visible=False) # explain_output placeholder
        ]

        demo.load(fetch_full_state, outputs=all_outputs)
        reset_btn.click(on_reset, inputs=[task_dropdown], outputs=all_outputs)
        step_btn.click(on_step, inputs=[tool_dropdown, params_input, reasoning_input], outputs=all_outputs)
        grade_btn.click(on_grade, outputs=[eval_output])
        
        def update_params(tool):
            if tool == "extract_ioc":
                try:
                    with httpx.Client(timeout=5) as client:
                        st = client.get(f"{server_url}/state").json()
                    text = st.get("logs", "") + "\n" + st.get("code_snippet", "")
                    cands = list(dict.fromkeys(re.findall(r'sk_(?:live|test)_\w{10,}|(?:\d{1,3}\.){3}\d{1,3}|[\w-]+\.(?:xyz|cc|ru|tk|onion|io)', text)))
                    return gr.update(choices=cands, value=cands[0] if cands else "")
                except: return gr.update(choices=[])
            
            targets = {
                "query_logs": ["all", "auth.log", "access.log", "error.log", "system.log"],
                "inspect_file": ["app.log", "config.py", "vendor/auth_lib.py", "requirements.txt", "index.html", "auth_service.py"],
                "apply_fix": ["remediate"]
            }
            return gr.update(choices=targets.get(tool, []), value=targets.get(tool, [""])[0])

        tool_dropdown.change(update_params, inputs=[tool_dropdown], outputs=[params_input])

    return demo
