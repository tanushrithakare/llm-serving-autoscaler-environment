import gradio as gr
import httpx
import os

CSS = """
.gradio-container {
    font-family: 'Segoe UI', 'Inter', system-ui, sans-serif !important;
    max-width: 1400px !important;
}

/* Terminal-style code blocks */
.code-wrap textarea, .code-wrap code {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
    font-size: 0.82em !important;
    line-height: 1.5 !important;
}

/* Base status card — charcoal, not black */
.hud-card {
    background: #2c2c2c;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 6px;
    color: #e8e8e8;
    font-size: 0.92em;
}

/* Success — green accent only on border + text, not full green background */
.hud-card-success {
    background: #2a2a2a;
    border: 1px solid #3a7d44;
    color: #7dcf8e;
}

/* Danger — muted red tint, not saturated */
.hud-card-danger {
    background: #2e2020;
    border: 1px solid #7a2a2a;
    color: #f2a0a0;
}

/* Warning — muted amber tint */
.hud-card-warn {
    background: #2c2616;
    border: 1px solid #8a6a20;
    color: #e0b84a;
}

/* Alert banner — dark charcoal-red, not pure black */
.alert-banner {
    background: #2e1515;
    border-left: 4px solid #cc3333;
    border-radius: 4px;
    padding: 10px 16px;
    margin-bottom: 10px;
    color: #f0c0c0;
    font-size: 0.95em;
    letter-spacing: 0.02em;
}
"""

def create_gradio_ui(server_url: str = "http://localhost:7860"):

    # ── Helper functions (unchanged logic) ──────────────────────────────────

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

        # Dynamic status signal
        if raw_status == "Mitigated":
            status_display = "🟢 Mitigated"
        elif root_cause or ioc:
            status_display = "🟡 Threat Identified — Remediation Pending"
        else:
            status_display = "🔴 Threat Ongoing — Investigation Active"

        # Action Taken: reflects real progress, not a static label
        if action_taken:
            action_display = "🟢 Remediation Applied"
        elif len(history) > 0:
            action_display = "🟡 Investigation in Progress"
        else:
            action_display = "⚠️ Pending"

        # Confidence: only shown after both root cause + IOC are confirmed
        confidence_display = f"{confidence:.2f}" if (root_cause and ioc) else "—"

        def fmt(val):
            return f"`{val}`" if val else "⚠️ Pending"

        # Investigation Behavior: Signals adaptive reasoning vs efficiency
        has_mistakes = any(h.get("reward", 0) < 0 for h in history)
        if has_mistakes:
            behavior_signal = "⚠️ **Initial missteps detected** — Analyst adjusted strategy and pivoted to correct artifacts."
        elif len(history) > 0:
            behavior_signal = "✅ **High-precision investigation** — Analyst correctly prioritized findings with zero wasted actions."
        else:
            behavior_signal = "—"

        # No '### header' here — the tab label already shows '🧾 Incident Report'
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
        title="Sentinel-SOC: AI Security Analyst",
        css=CSS,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    ) as demo:

        # ── TOP SECTION ──────────────────────────────────────────────────────
        gr.Markdown("# 🛡️ SENTINEL-SOC: AI Security Analyst System")
        gr.HTML(
            '<div class="alert-banner">'
            '⚠️ &nbsp;<strong>LIVE INCIDENT: Potential Security Breach Detected</strong>'
            ' — Investigation in progress'
            '</div>'
        )
        gr.HTML(
            '<p style="margin:2px 0 8px 0; font-size:0.76em; color:#777; letter-spacing:0.06em;">'
            '🟢 Operational &nbsp;•&nbsp; Analyst Active &nbsp;•&nbsp; Kill Chain Enabled'
            '</p>'
        )
        gr.Markdown("---")

        # ── MAIN LAYOUT: left control panel + right investigation area ────────
        with gr.Row(equal_height=False):

            # ── LEFT: Control Panel ──────────────────────────────────────────
            with gr.Column(scale=1, min_width=240):

                gr.Markdown("### 🖥️ System Status")
                status_box   = gr.Markdown("**Status**: Active",      elem_classes=["hud-card"])
                severity_box = gr.Markdown("**Severity**: 🟠 Medium", elem_classes=["hud-card-warn"])
                steps_box    = gr.Markdown("**Steps Remaining**: 20", elem_classes=["hud-card"])
                reward_box   = gr.Markdown("### 📊 Score: `0.00`",  elem_classes=["hud-card"])

                gr.Markdown("---")
                gr.Markdown("### ⚙️ Scenario")
                task_dropdown = gr.Radio(
                    choices=["easy", "medium", "hard"],
                    value="easy",
                    label="Difficulty"
                )
                reset_btn = gr.Button("🔄 Initialize", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### 🤖 Agent Action Console")
                gr.HTML(
                    '<p style="font-size:0.76em; color:#999; margin:0 0 8px 0;">'
                    'Simulate how an AI security analyst selects tools and investigates artifacts step-by-step.'
                    '</p>'
                )
                tool_dropdown = gr.Dropdown(
                    choices=["query_logs", "extract_ioc", "inspect_file", "apply_fix"],
                    value="query_logs",
                    label="Action"
                )
                params_input = gr.Dropdown(
                    choices=["all", "auth.log", "access.log", "error.log", "system.log"],
                    value="all",
                    label="Target",
                    allow_custom_value=True
                )
                gr.HTML(
                    '<p style="font-size:0.72em; color:#777; margin:-4px 0 8px 0;">'
                    '🖥️ Targets sourced from current system state and telemetry'
                    '</p>'
                )
                reasoning_input = gr.Textbox(
                    label="Analyst Thought",
                    placeholder="Explain the investigative reasoning behind this action...",
                    lines=2
                )
                step_btn  = gr.Button("▶ Execute Action",        variant="secondary")
                gr.Markdown("---")
                grade_btn = gr.Button("📊 Evaluate Investigation", variant="stop")

            # ── RIGHT: Investigation Area ────────────────────────────────────
            with gr.Column(scale=3):
                with gr.Tabs():

                    with gr.TabItem("📄 Logs"):
                        gr.Markdown("## 📡 Live Telemetry Feed")
                        logs_output = gr.Code(
                            label="",
                            language="shell",
                            lines=20,
                            interactive=False
                        )

                    with gr.TabItem("💻 Source"):
                        code_output = gr.Code(
                            label="Source Code",
                            language="python",
                            lines=22,
                            interactive=False
                        )

                    with gr.TabItem("🕵️ Intel"):
                        thread_output = gr.Markdown(label="Threat Assessment")

                    with gr.TabItem("📜 Timeline"):
                        gr.Markdown("##### 🕵️ Investigation Timeline")
                        timeline_output = gr.Markdown("No steps recorded yet.")

                    with gr.TabItem("🧠 Reasoning"):
                        gr.Markdown("##### 🧠 Analyst Reasoning Trace")
                        reasoning_output = gr.Markdown("Execute a step to see agent thinking.")

                    with gr.TabItem("📊 Summary"):
                        gr.Markdown("##### 🧾 Incident Report")
                        summary_output = gr.Markdown("Investigation has not started.")

                    with gr.TabItem("🏆 Evaluation"):
                        gr.Markdown("##### 🏆 Kill Chain Score Breakdown")
                        eval_output = gr.Markdown("Complete the investigation to see evaluation.")

                    with gr.TabItem("💡 Explain"):
                        gr.Markdown("##### 💡 Plain-English Briefing")
                        explain_output = gr.Markdown("Start the investigation to generate an explanation.")

        # ── State management (unchanged logic) ───────────────────────────────
        def fetch_full_state():
            try:
                with httpx.Client(timeout=15) as client:
                    resp = client.get(f"{server_url}/state")
                    if resp.status_code != 200:
                        return ["Error fetching state"] * 12
                    data         = resp.json()
                    history_resp = client.get(f"{server_url}/history")
                    history_data = history_resp.json().get("history", [])
                    max_steps    = data.get("steps_remaining", 0) + len(history_data)
                    return [
                        f"**Status**: {data.get('status', 'Active')}",
                        f"**Severity**: {build_severity(data)}",
                        f"**Steps Remaining**: {data.get('steps_remaining', 0)}",
                        f"### 📊 Score: `{data.get('reward_signal', 0.0):.2f}`",
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
                    "reasoning":  reasoning or f"Executing {tool}",
                    "tool":       tool,
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
                    score_resp   = client.post(f"{server_url}/grade")
                    score        = score_resp.json().get("score", 0)
                    history_resp = client.get(f"{server_url}/history")
                    history_data = history_resp.json().get("history", [])
                    state_resp   = client.get(f"{server_url}/state")
                    state_data   = state_resp.json()
                    max_steps    = state_data.get("steps_remaining", 0) + len(history_data)
                    return build_evaluation(score, history_data, max_steps)
            except Exception as e:
                return f"ERR: {str(e)}"

        # ── Wire events ───────────────────────────────────────────────────────
        all_outputs = [
            status_box, severity_box, steps_box, reward_box,
            logs_output, code_output, thread_output,
            timeline_output, reasoning_output, summary_output,
            eval_output, explain_output,
        ]

        reset_btn.click(on_reset, inputs=[task_dropdown],                               outputs=all_outputs)
        step_btn.click( on_step,  inputs=[tool_dropdown, params_input, reasoning_input],  outputs=all_outputs)
        grade_btn.click(on_grade,                                                       outputs=[eval_output])
        demo.load(fetch_full_state,                                                     outputs=all_outputs)

        import re

        TOOL_TARGETS = {
            "query_logs":   ["all", "auth.log", "access.log", "error.log", "system.log"],
            "inspect_file": ["app.log", "config.py", "vendor/auth_lib.py",
                             "requirements.txt", "index.html", "auth_service.py"],
            "apply_fix":    ["remediate"],
        }

        def extract_ioc_candidates():
            """Parse live logs + code_snippet for IOC candidates."""
            try:
                with httpx.Client(timeout=5) as client:
                    data = client.get(f"{server_url}/state").json()
                text = data.get("logs", "") + "\n" + data.get("code_snippet", "")
                candidates = []
                # API keys: sk_live_... / sk_test_...
                candidates += re.findall(r'sk_(?:live|test)_\w{10,}', text)
                # IPv4 addresses (non-local)
                ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text)
                candidates += [ip for ip in ips if not ip.startswith(("127.", "10.", "192.168.", "0."))]
                # Suspicious domains (.xyz .cc .ru .tk .onion)
                candidates += re.findall(r'\b[\w-]+\.(?:xyz|cc|ru|tk|onion|io)\b', text)
                # Base64-looking strings (20+ chars)
                candidates += re.findall(r'[A-Za-z0-9+/]{20,}={0,2}', text)[:3]
                # Deduplicate, preserve order
                seen, unique = set(), []
                for c in candidates:
                    if c not in seen:
                        seen.add(c)
                        unique.append(c)
                return unique or ["— run query_logs first —"]
            except Exception:
                return ["— run query_logs first —"]

        def update_targets(tool):
            if tool == "extract_ioc":
                targets = extract_ioc_candidates()
            else:
                targets = TOOL_TARGETS.get(tool, [])
            first = targets[0] if targets else ""
            return gr.Dropdown(choices=targets, value=first, allow_custom_value=True)

        tool_dropdown.change(update_targets, inputs=[tool_dropdown], outputs=[params_input])

    return demo
