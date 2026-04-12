import gradio as gr
import httpx
import os
import re
import json
import base64
import numpy as np

CSS = """
.gradio-container {
    font-family: 'Segoe UI', 'Inter', system-ui, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}

/* Hard-coded Dark Body - for safety */
body { background-color: #111 !important; color: #eee !important; }

/* Terminal-style code blocks */
.code-wrap textarea, .code-wrap code {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
    font-size: 0.82em !important;
    line-height: 1.5 !important;
}

/* HUB Card - Compact Version */
.hud-card {
    background: #222 !important;
    border: 1px solid #444 !important;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px;
    color: #eee !important;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.5);
}

.hud-card-success { border-color: #22c55e !important; color: #4ade80 !important; box-shadow: 0 0 10px rgba(34,197,94,0.2) !important; }
.hud-card-danger { border-color: #ef4444 !important; color: #f87171 !important; box-shadow: 0 0 10px rgba(239,68,68,0.2) !important; }
.hud-card-warn { border-color: #f59e0b !important; color: #fbbf24 !important; box-shadow: 0 0 10px rgba(245,158,11,0.2) !important; }
.hud-card-info { border-color: #3b82f6 !important; color: #60a5fa !important; box-shadow: 0 0 10px rgba(59,130,246,0.2) !important; }

/* Alert banner */
.alert-banner {
    background: #2e1515;
    border-left: 4px solid #ef4444;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 5px 0 15px 0;
    color: #fca5a5;
    font-size: 0.9em;
    font-weight: 500;
}

/* Action Workbench styling */
.workbench-group {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 10px !important;
    padding: 15px !important;
}

.primary-btn {
    background-color: #3b82f6 !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
}

.secondary-btn {
    background-color: #334155 !important;
    color: #e2e8f0 !important;
    border: none !important;
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
            params   = h.get("params", "")
            reward   = h.get("reward", 0)
            feedback = h.get("feedback", "")
            icon     = "🟢" if reward > 0 else "🔴" if reward < 0 else "🟡"
            sign     = "+" if reward > 0 else ""
            lines.append(
                f"**Step {step} — `{tool}`** {icon}\n\n"
                f"📄 Target: `{params}`  \n"
                f"{'✅' if reward > 0 else '⚠️'} Result: {feedback}  \n"
                f"📉 Reward: `{sign}{reward:.2f}`\n"
                f"---"
            )
        return "\n".join(lines)

    def build_reasoning(history):
        if not history:
            return "No reasoning recorded yet. Execute a step to see agent thinking."
        lines = []
        for h in history:
            step      = h.get("step", "?")
            tool      = h.get("tool", "")
            reasoning = h.get("reasoning", "No reasoning provided")
            reward    = h.get("reward", 0)
            icon      = "🟢" if reward > 0 else "🔴" if reward < 0 else "🟡"
            lines.append(
                f"### Step {step} — `{tool}` {icon}\n"
                f"**Thought:** {reasoning}\n"
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
            return "Investigation has not started."

        root_cause   = None
        ioc          = None
        action_taken = None

        for h in history:
            if h.get("tool") == "inspect_file" and h.get("reward", 0) > 0:
                root_cause = h.get("params", "Unknown")
            if h.get("tool") == "extract_ioc" and h.get("reward", 0) > 0:
                ioc = h.get("params", "Unknown")
            if h.get("tool") == "apply_fix" and h.get("reward", 0) > 0:
                action_taken = True

        confidence = sum(h.get("confidence", 0) for h in history) / max(len(history), 1)
        raw_status = state_data.get("status", "Active") if state_data else "Active"
        max_steps  = (state_data.get("steps_remaining", 0) if state_data else 0) + len(history)

        if raw_status == "Mitigated":
            status_display = "🟢 Mitigated"
        elif root_cause or ioc:
            status_display = "🟡 Threat Identified — Remediation Pending"
        else:
            status_display = "🔴 Threat Ongoing — Investigation Active"

        if action_taken:
            action_display = "🟢 Remediation Applied"
        elif len(history) > 0:
            action_display = "🟡 Investigation in Progress"
        else:
            action_display = "⚠️ Pending"

        confidence_display = f"{confidence:.2f}" if (root_cause and ioc) else "—"

        def fmt(val):
            return f"`{val}`" if val else "⚠️ Pending"

        # Behavior Signal: Evidence of adaptive intelligence
        has_mistakes = any(h.get("reward", 0) < 0 for h in history)
        if has_mistakes:
            behavior_signal = "⚠️ **Initial missteps detected** — Analyst adjusted strategy and pivoted to correct artifacts."
        elif len(history) > 0:
            behavior_signal = "✅ **High-precision investigation** — Analyst correctly prioritized findings with zero wasted actions."
        else:
            behavior_signal = "—"

        return (
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| **Status** | {status_display} |\n"
            f"| **Root Cause** | {fmt(root_cause)} |\n"
            f"| **IOC** | {fmt(ioc)} |\n"
            f"| **Action Taken** | {action_display} |\n"
            f"| **Confidence** | {confidence_display} |\n"
            f"| **Steps Used** | {len(history)} / {max_steps} |\n\n"
            f"**🧠 Investigation Behavior**\n\n"
            f"{behavior_signal}"
        )


    def build_evaluation(score, history, max_steps):
        phases = {
            "Reconnaissance": {"done": False, "reward": 0},
            "Identification":  {"done": False, "reward": 0},
            "Containment":     {"done": False, "reward": 0},
            "Remediation":     {"done": False, "reward": 0},
        }
        mistakes = []
        for h in history:
            tool   = h.get("tool", "")
            reward = h.get("reward", 0)
            if tool == "query_logs"  and reward > 0: phases["Reconnaissance"] = {"done": True, "reward": reward}
            elif tool == "extract_ioc" and reward > 0: phases["Identification"]  = {"done": True, "reward": reward}
            elif tool == "inspect_file" and reward > 0: phases["Containment"]    = {"done": True, "reward": reward}
            elif tool == "apply_fix"    and reward > 0: phases["Remediation"]    = {"done": True, "reward": reward}
            if reward < 0:
                mistakes.append(f"Step {h.get('step')}: {h.get('feedback', 'Error')} ({reward:+.2f})")
        
        lines = ["### Kill Chain Evaluation\n"]
        for phase, data in phases.items():
            icon       = "✅" if data["done"] else "⬜"
            reward_str = f"+{data['reward']:.2f}" if data["done"] else "pending"
            lines.append(f"{icon} **{phase}** — {reward_str}")
        
        efficiency = len(history) / max(max_steps, 1)
        penalty    = efficiency * 0.15
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
        if not state_data:
            return "UNKNOWN"
        status    = state_data.get("status", "Active")
        remaining = state_data.get("steps_remaining", 0)
        if status == "Mitigated":          return "🟢 Resolved"
        elif remaining <= 3:               return "🔴 Critical"
        elif remaining <= 8:               return "🟡 High"
        else:                              return "🟠 Medium"

    # ── Build UI ─────────────────────────────────────────────────────────────
    with gr.Blocks(
        title="Sentinel-SOC | AI Security Analyst",
        css=CSS,
        theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate").set(
            body_background_fill="#111",
            block_background_fill="#1a1a1a",
            block_border_width="1px",
            block_title_text_color="#eee"
        )
    ) as demo:

        # ── HEADER ───────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=8):
                gr.Markdown("# 🛡️ SENTINEL-SOC COMMAND CENTER")
                gr.HTML(
                    '<div class="alert-banner">'
                    '⚠️ &nbsp;<strong>THREAT ALERT: ACTIVE INCIDENT DETECTED</strong>'
                    ' &nbsp;—&nbsp; Analyzing System Artifacts'
                    '</div>'
                )
            with gr.Column(scale=2, min_width=150):
                gr.HTML(
                    '<p style="margin-top:20px; text-align:right; font-size:0.75em; color:#666;">'
                    '🟢 SYSTEM OPERATIONAL<br>🔐 KILL-CHAIN ACTIVE'
                    '</p>'
                )

        # ── HUD (Top Hero Metrics Row) ───────────────────────────────────────
        with gr.Row():
            gr.Column(scale=1) # Spacer
            with gr.Column(scale=2, min_width=150):
                status_box   = gr.HTML(label="Status")
            with gr.Column(scale=2, min_width=150):
                severity_box = gr.HTML(label="Severity")
            with gr.Column(scale=2, min_width=150):
                steps_box    = gr.HTML(label="Progress")
            with gr.Column(scale=2, min_width=150):
                score_box    = gr.HTML(label="Score")
            gr.Column(scale=1) # Spacer

        gr.Markdown("---")

        with gr.Row(equal_height=False):

            # ── LEFT: RECON WORKBENCH ────────────────────────────────────────
            with gr.Column(scale=4):
                with gr.Group(elem_classes=["workbench-group"]):
                    gr.Markdown("### 🛠️ Analyst Action Workbench")
                    
                    with gr.Row():
                        task_dropdown = gr.Radio(
                            choices=["easy", "medium", "hard"],
                            value="easy",
                            label="Incident Severity (Level)"
                        )
                        reset_btn = gr.Button("🔄 Reset Scenario", variant="secondary", size="sm", elem_classes=["secondary-btn"])

                    gr.Markdown("---")
                    
                    with gr.Row():
                        tool_dropdown = gr.Dropdown(
                            choices=["query_logs", "extract_ioc", "inspect_file", "apply_fix"],
                            value="query_logs",
                            label="Forensic Tool"
                        )
                        params_input = gr.Dropdown(
                            choices=["all", "auth.log", "access.log", "error.log", "system.log"],
                            value="all",
                            label="Target Parameter",
                            allow_custom_value=True
                        )
                    
                    reasoning_input = gr.Textbox(
                        label="Investigative Reasoning",
                        placeholder="Explain why this action is necessary for the current kill-chain phase...",
                        lines=2
                    )
                    
                    with gr.Row():
                        step_btn = gr.Button("▶ Execute Action", variant="primary", scale=2, elem_classes=["primary-btn"])
                        grade_btn = gr.Button("📊 Finalize & Grade", variant="stop", scale=1)

            # ── RIGHT: TELEMETRY & INTEL ─────────────────────────────────────
            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.TabItem("📊 SUMMARY"):
                        gr.Markdown("##### 🧾 Incident Report Summary")
                        summary_output = gr.Markdown("Waiting for investigation initialization...")

                    with gr.TabItem("🕵️ LOGS"):
                        gr.Markdown("##### 📡 Live System Telemetry")
                        logs_output = gr.Code(label="", language="shell", lines=18, interactive=False)

                    with gr.TabItem("💻 SOURCE"):
                        gr.Markdown("##### 💻 Suspicious Source Investigation")
                        code_output = gr.Code(label="", language="python", lines=18, interactive=False)

                    with gr.TabItem("📜 TIMELINE"):
                        gr.Markdown("##### 📜 Step-by-Step Investigation History")
                        timeline_output = gr.Markdown("Activity log is empty.")

                    with gr.TabItem("🧠 BRAIN"):
                        gr.Markdown("##### 🧠 Agent Reasoning Trace")
                        reasoning_output = gr.Markdown("Execute actions to see thoughts.")

                    with gr.TabItem("🏆 GRADE"):
                        gr.Markdown("##### 🏆 Evaluation & Kill-Chain Breakdown")
                        eval_output = gr.Markdown("Complete investigation to generate final report.")

                    with gr.TabItem("💡 EXPLAIN"):
                        gr.Markdown("##### 💡 Natural Language Briefing")
                        explain_output = gr.Markdown("Explain the incident in plain English.")

        # ── STATE LOGIC ──────────────────────────────────────────────────────
        def fmt_hud(label, value, status="neutral"):
            """Helper to generate premium HTML HUD cards with dynamic colors."""
            class_map = {
                "success": "hud-card-success",
                "danger": "hud-card-danger",
                "warn": "hud-card-warn",
                "info": "hud-card-info",
                "neutral": "hud-card"
            }
            cls = class_map.get(status, "hud-card")
            return f'<div class="hud-card {cls}"><p style="margin:0; font-size:0.8em; opacity:0.8; font-weight:600;">{label}</p><h3 style="margin:5px 0 0 0; font-size:1.6em;">{value}</h3></div>'

        def fetch_full_state():
            try:
                with httpx.Client(timeout=15) as client:
                    resp = client.get(f"{server_url}/state")
                    if resp.status_code != 200:
                        return ["ERR"] * 11
                    data = resp.json()
                    history_resp = client.get(f"{server_url}/history")
                    history_data = history_resp.json().get("history", [])
                    max_steps = data.get("steps_remaining", 0) + len(history_data)
                    
                    # 1. Status Card (Grey by default, Green on Success)
                    curr_status = data.get('status', 'Active')
                    stat_cls = "success" if curr_status == "Mitigated" else "neutral"
                    status_html = fmt_hud("SYSTEM STATUS", curr_status.upper(), stat_cls)
                    
                    # 2. Severity Card (Grey baseline, Red for Critical, Yellow for High)
                    sev_text = build_severity(data).split(' ')[1].upper()
                    sev_cls = "danger" if sev_text == "CRITICAL" else ("warn" if sev_text == "HIGH" else "neutral")
                    severity_html = fmt_hud("THREAT LEVEL", sev_text, sev_cls)
                    
                    # 3. Progress Card (Clean Neutral Grey)
                    progress_html = fmt_hud("INVESTIGATION PROGRESS", f"{len(history_data)} / {max_steps}", "neutral")
                    
                    # 4. Score Card (Info Blue)
                    score_val = data.get('reward_signal', 0.0)
                    score_html = fmt_hud("EFFICIENCY SCORE", f"{score_val:.2f}", "info")
                    
                    return [
                        status_html,
                        severity_html,
                        progress_html,
                        score_html,
                        data.get("logs", ""),
                        data.get("code_snippet", ""),
                        build_timeline(history_data),
                        build_reasoning(history_data),
                        build_summary(history_data, data),
                        build_evaluation(None, history_data, max_steps),
                        build_explanation(history_data, data),
                    ]
            except Exception as e:
                return [f"ERR: {str(e)}"] * 11

        def on_reset(task):
            try:
                with httpx.Client(timeout=15) as client:
                    client.post(f"{server_url}/reset", params={"task": task})
                return fetch_full_state()
            except Exception:
                return ["ERR"] * 11

        def on_step(tool, params, reasoning):
            try:
                action = {"reasoning": reasoning or f"Evaluating {tool}", "tool": tool, "parameters": params or ""}
                with httpx.Client(timeout=15) as client:
                    client.post(f"{server_url}/step", json=action)
                return fetch_full_state()
            except Exception:
                return ["ERR"] * 11

        def on_grade():
            try:
                with httpx.Client(timeout=15) as client:
                    score = client.post(f"{server_url}/grade").json().get("score", 0)
                    history = client.get(f"{server_url}/history").json().get("history", [])
                    state = client.get(f"{server_url}/state").json()
                    max_steps = state.get("steps_remaining", 0) + len(history)
                    return build_evaluation(score, history, max_steps)
            except Exception:
                return "ERR"

        # ── DYNAMIC TARGET EXTRACTION ────────────────────────────────────────
        TOOL_TARGETS = {
            "query_logs": ["all", "auth.log", "access.log", "error.log", "system.log"],
            "inspect_file": ["app.log", "config.py", "vendor/auth_lib.py", "requirements.txt", "index.html", "auth_service.py"],
            "apply_fix": ["remediate"],
        }

        def update_targets(tool):
            if tool == "extract_ioc":
                # Scrape current logs/code for IOCs via the server state
                try:
                    with httpx.Client(timeout=5) as client:
                        data = client.get(f"{server_url}/state").json()
                    text = data.get("logs", "") + "\n" + data.get("code_snippet", "")
                    candidates = []
                    candidates += re.findall(r'sk_(?:live|test)_\w{10,}', text)
                    ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text)
                    candidates += [ip for ip in ips if not ip.startswith(("127.", "10.", "192.168.", "0."))]
                    candidates += re.findall(r'\b[\w-]+\.(?:xyz|cc|ru|tk|onion|io)\b', text)
                    unique = list(dict.fromkeys(candidates))
                    return gr.Dropdown(choices=unique or ["— run query_logs first —"], value=unique[0] if unique else "", allow_custom_value=True)
                except Exception:
                    return gr.Dropdown(choices=["— run query_logs first —"], value="")
            
            targets = TOOL_TARGETS.get(tool, [])
            return gr.Dropdown(choices=targets, value=targets[0] if targets else "", allow_custom_value=True)

        # ── WIRING ───────────────────────────────────────────────────────────
        all_outputs = [
            status_box, severity_box, steps_box, score_box,
            logs_output, code_output,
            timeline_output, reasoning_output, summary_output,
            eval_output, explain_output,
        ]

        reset_btn.click(on_reset, inputs=[task_dropdown], outputs=all_outputs)
        step_btn.click(on_step, inputs=[tool_dropdown, params_input, reasoning_input], outputs=all_outputs)
        grade_btn.click(on_grade, outputs=[eval_output])
        tool_dropdown.change(update_targets, inputs=[tool_dropdown], outputs=[params_input])
        demo.load(fetch_full_state, outputs=all_outputs)

    return demo
