import gradio as gr
import httpx
import os

# Custom CSS for that high-tech Forensic HUD look
CSS = """
.gradio-container {
    background-color: #0b0e14 !important;
    color: #00ff41 !important;
    font-family: 'Courier New', Courier, monospace !important;
}
.sidebar {
    border-right: 1px solid #00ff41 !important;
    padding-right: 20px;
}
.status-active {
    color: #ff9d00 !important;
    font-weight: bold;
    animation: blink 2s infinite;
}
@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
.hud-card {
    background: rgba(0, 255, 65, 0.05);
    border: 1px solid #00ff41;
    border-radius: 4px;
    padding: 10px;
    margin-bottom: 10px;
}
"""

def create_gradio_ui(server_url: str = "http://localhost:7860"):
    with gr.Blocks(title="Sentinel-SOC Forensic Dashboard", css=CSS, theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate")) as demo:
        gr.Markdown("# 🛡️ SENTINEL-SOC: Forensic Dashboard")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 INCIDENT HUD")
                status_box = gr.Markdown("STAUTS: **ACTIVE**", elem_classes=["hud-card", "status-active"])
                reward_box = gr.Markdown("EFFICIENCY: **0.00**", elem_classes=["hud-card"])
                steps_box = gr.Markdown("TTL: **10**", elem_classes=["hud-card"])
                
                gr.Markdown("### 🛠️ OPERATIONS")
                task_dropdown = gr.Radio(
                    choices=["easy", "medium", "hard"], 
                    value="easy", 
                    label="Deployment Scenario"
                )
                reset_btn = gr.Button("🔄 INITIALIZE ENVIRONMENT", variant="primary")
            
            with gr.Column(scale=3):
                gr.Markdown("### 🔬 FORENSIC DATA")
                with gr.Tabs():
                    with gr.TabItem("📄 System Logs"):
                        logs_output = gr.Code(label="Live Logs", language=None, lines=15, interactive=False)
                    with gr.TabItem("💻 Source Inspector"):
                        code_output = gr.Code(label="Target Source Code", language="python", lines=15, interactive=False)
                    with gr.TabItem("🕵️ Investigation Thread"):
                        thread_output = gr.Markdown(label="Incident Intel")
        
        with gr.Row():
            gr.Markdown("### 📜 INVESTIGATION TIMELINE")
            history_table = gr.Dataframe(
                headers=["Step", "Tool", "Parameters", "Feedback", "Confidence"],
                datatype=["number", "str", "str", "str", "number"],
                column_widths=[50, 150, 250, 350, 100],
                interactive=False
            )

        def update_state():
            try:
                with httpx.Client() as client:
                    resp = client.get(f"{server_url}/state")
                    if resp.status_code == 200:
                        data = resp.json()
                        history_resp = client.get(f"{server_url}/history")
                        history_data = history_resp.json().get("history", [])
                        
                        # Format history for dataframe
                        df_data = []
                        for h in history_data:
                            df_data.append([
                                h.get("step"),
                                h.get("tool"),
                                h.get("params"),
                                h.get("feedback"),
                                h.get("confidence")
                            ])
                            
                        return [
                            f"STATUS: **{data.get('status')}**",
                            f"EFFICIENCY: **{data.get('reward_signal')}**",
                            f"TTL: **{data.get('steps_remaining')}**",
                            data.get("logs"),
                            data.get("code_snippet"),
                            data.get("incident_thread"),
                            df_data
                        ]
            except Exception as e:
                return [f"ERR: {str(e)}"] * 7
            return ["No data"] * 7

        def on_reset(task):
            try:
                with httpx.Client() as client:
                    client.post(f"{server_url}/reset", params={"task": task})
                return update_state()
            except Exception as e:
                return [f"ERR: {str(e)}"] * 7

        # Event handlers
        reset_btn.click(on_reset, inputs=[task_dropdown], outputs=[status_box, reward_box, steps_box, logs_output, code_output, thread_output, history_table])
        
        # Load initial state
        demo.load(update_state, outputs=[status_box, reward_box, steps_box, logs_output, code_output, thread_output, history_table])
        
    return demo
