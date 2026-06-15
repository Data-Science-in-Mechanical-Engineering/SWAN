#!/usr/bin/env python3
"""Gradio UI for SWAN drone show pipeline with SSH execution."""

import gradio as gr
import paramiko
import os
import subprocess
import tempfile
import shutil
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SSHConfig:
    host: str = "localhost"
    port: int = 22
    username: str = ""
    password: str = ""
    remote_output_dir: str = "/app/out"
    container_name: str = "swan"


@dataclass
class PipelineState:
    ssh_config: SSHConfig = field(default_factory=SSHConfig)
    mount_path: Optional[str] = None
    ssh_client: Optional[paramiko.SSHClient] = None
    ssh_connected: bool = False
    output_dir_name: str = ""
    current_stage: int = 0
    n_drones: int = 500
    user_prompt: str = ""
    stage_outputs: dict = field(default_factory=dict)
    logs: str = ""


def ssh_connect(ssh_config: SSHConfig) -> paramiko.SSHClient:
    """Establishes SSH connection using paramiko."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=ssh_config.host,
        port=ssh_config.port,
        username=ssh_config.username,
        password=ssh_config.password,
        timeout=30
    )
    return client


def mount_remote_directory(ssh_config: SSHConfig, local_mount_base: str) -> str:
    """Mount remote output directory via sshfs."""
    mount_path = os.path.join(local_mount_base, f"swan_mount_{int(time.time())}")
    os.makedirs(mount_path, exist_ok=True)
    
    if ssh_config.password:
        mount_cmd = f"sshpass -p '{ssh_config.password}' sshfs {ssh_config.username}@{ssh_config.host}:{ssh_config.remote_output_dir} {mount_path}"
    else:
        mount_cmd = f"sshfs {ssh_config.username}@{ssh_config.host}:{ssh_config.remote_output_dir} {mount_path}"
    
    try:
        result = subprocess.run(mount_cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"SSHFS mount warning: {result.stderr}")
    except Exception as e:
        print(f"SSHFS mount error: {e}")
    
    return mount_path


def unmount_remote_directory(mount_path: Optional[str]):
    """Unmount sshfs directory."""
    if mount_path and os.path.exists(mount_path):
        try:
            subprocess.run(f"fusermount -u {mount_path}", shell=True, timeout=30)
            shutil.rmtree(mount_path, ignore_errors=True)
        except Exception:
            pass


def run_remote_command(ssh_client, command: str) -> tuple[int, str]:
    """Execute command on remote server and stream output."""
    stdin, stdout, stderr = ssh_client.exec_command(command)
    exit_status = stdout.channel.get_exit_status()
    
    output_lines = []
    while not stdout.channel.exit_status_ready():
        line = stdout.readline()
        if line:
            output_lines.append(line)
    
    remaining = stdout.read().decode()
    if remaining:
        output_lines.append(remaining)
    
    stderr_output = stderr.read().decode()
    if stderr_output:
        output_lines.append(f"STDERR: {stderr_output}")
    
    return exit_status, "".join(output_lines)


def run_pipeline_stage(stage: str, ssh_client, ssh_config: SSHConfig, output_dir_name: str, n_drones: int, user_prompt: str, mount_path: str) -> tuple[bool, str]:
    """Run a specific pipeline stage on the remote server."""
    safe_prompt = user_prompt.replace("'", "'\"'\"'")
    
    if stage == "video":
        cmd = f'docker exec {ssh_config.container_name} python /app/main.py --n-drones {n_drones} --video-name "{output_dir_name}" --user-prompt \'{safe_prompt}\' --output-dir {ssh_config.remote_output_dir}/{output_dir_name}_{n_drones}_drones --no-prompt'
    elif stage == "tracking":
        cmd = f'docker exec {ssh_config.container_name} python /app/main.py --n-drones {n_drones} --video-name "{output_dir_name}" --start-stage tracking --no-prompt --output-dir {ssh_config.remote_output_dir}/{output_dir_name}_{n_drones}_drones'
    elif stage == "trajectory":
        cmd = f'docker exec {ssh_config.container_name} python /app/main.py --n-drones {n_drones} --video-name "{output_dir_name}" --start-stage trajectory --no-prompt --output-dir {ssh_config.remote_output_dir}/{output_dir_name}_{n_drones}_drones'
    elif stage == "simulation":
        cmd = f'docker exec {ssh_config.container_name} python /app/main.py --n-drones {n_drones} --video-name "{output_dir_name}" --start-stage simulation --no-prompt --output-dir {ssh_config.remote_output_dir}/{output_dir_name}_{n_drones}_drones'
    elif stage == "visualization":
        cmd = f'docker exec {ssh_config.container_name} python /app/main.py --n-drones {n_drones} --video-name "{output_dir_name}" --start-stage visualization --no-prompt --output-dir {ssh_config.remote_output_dir}/{output_dir_name}_{n_drones}_drones'
    else:
        return False, f"Unknown stage: {stage}"
    
    try:
        exit_status, output = run_remote_command(ssh_client, cmd)
        return exit_status == 0, output
    except Exception as e:
        return False, str(e)


def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks() as demo:
        pipeline_state = gr.State(PipelineState())
        
        with gr.Accordion("Configuration", open=True):
            show_name = gr.Textbox(label="Show Name", value="my_drone_show", placeholder="Enter a name for your drone show")
            user_prompt = gr.Textbox(
                label="User Prompt", 
                value="A T-Rex roaring and turning its head from left to right standing on a large flat rock.",
                lines=4,
                placeholder="Describe the drone show you want to create"
            )
            n_drones = gr.Slider(50, 2000, value=500, step=50, label="Number of Drones")
            
            with gr.Row():
                ssh_host = gr.Textbox(label="SSH Host", value="localhost")
                ssh_port = gr.Number(label="Port", value=22, precision=0)
            
            with gr.Row():
                ssh_username = gr.Textbox(label="Username")
                ssh_password = gr.Textbox(label="Password", type="password")
            
            with gr.Row():
                connect_btn = gr.Button("Connect", variant="primary")
                start_btn = gr.Button("Run Pipeline", variant="primary", visible=False)
                reset_btn = gr.Button("Reset", variant="secondary")
        
        with gr.Accordion("Stage 1: Video Generation", open=False):
            video_output = gr.Video(label="Generated Video", format="mp4")
            console1 = gr.Textbox(label="Console Output", lines=10, interactive=False)
            with gr.Row():
                continue1 = gr.Button("Continue →", visible=False)
                retry1 = gr.Button("Retry", variant="stop", visible=False)
        
        with gr.Accordion("Stage 2: Tracking", open=False):
            with gr.Row():
                tracking_video = gr.Video(label="Points on Video", format="mp4")
            with gr.Row():
                points_only_video = gr.Video(label="Points Only", format="mp4")
            console2 = gr.Textbox(label="Console Output", lines=10, interactive=False)
            with gr.Row():
                continue2 = gr.Button("Continue →", visible=False)
                retry2 = gr.Button("Retry", variant="stop", visible=False)
        
        with gr.Accordion("Stage 3: Trajectory Generation", open=False):
            console3 = gr.Textbox(label="Console Output", lines=10, interactive=False)
            with gr.Row():
                continue3 = gr.Button("Continue →", visible=False)
                retry3 = gr.Button("Retry", variant="stop", visible=False)
        
        with gr.Accordion("Stage 4: Simulation", open=False):
            console4 = gr.Textbox(label="Console Output", lines=10, interactive=False)
            with gr.Row():
                continue4 = gr.Button("Continue →", visible=False)
                retry4 = gr.Button("Retry", variant="stop", visible=False)
        
        with gr.Accordion("Stage 5: Final Show", open=False):
            assignment_video = gr.Video(label="Assignment View (Top-down)", format="mp4")
            tracking_overlay = gr.Video(label="Tracking Overlay", format="mp4")
            console5 = gr.Textbox(label="Console Output", lines=10, interactive=False)
        
        def connect_ssh(show_name_val, prompt_val, drones_val, host_val, port_val, user_val, pwd_val, state):
            """Establish SSH connection and mount remote directory."""
            state.ssh_config = SSHConfig(
                host=host_val,
                port=int(port_val),
                username=user_val,
                password=pwd_val
            )
            state.n_drones = int(drones_val)
            state.user_prompt = prompt_val
            state.output_dir_name = show_name_val.replace(" ", "_").replace("/", "_")
            
            try:
                ssh_client = ssh_connect(state.ssh_config)
                state.ssh_client = ssh_client
                state.mount_path = mount_remote_directory(state.ssh_config, tempfile.gettempdir())
                state.ssh_connected = True
                state.current_stage = 1
                state.logs = "SSH connection successful. Ready to run pipeline."
                # Show Run Pipeline button, hide Connect
                return state, gr.update(visible=False), gr.update(visible=True)
            except Exception as e:
                state.logs = f"SSH connection failed: {e}"
                state.ssh_connected = False
                return state, gr.update(visible=True), gr.update(visible=False)

        def run_video_stage(state):
            """Run the first pipeline stage (video generation) after SSH is ready."""
            if not state.ssh_connected or not state.ssh_client:
                state.logs = "SSH not connected."
                return state, None, None, "", gr.update(visible=False), gr.update(visible=True)
            try:
                success, output = run_pipeline_stage(
                    "video", state.ssh_client, state.ssh_config,
                    state.output_dir_name, state.n_drones, state.user_prompt,
                    state.mount_path
                )
                state.logs = output
                if success:
                    video_path = os.path.join(state.mount_path, f"{state.output_dir_name}_{state.n_drones}_drones", "video_gen", "video.mp4")
                    state.stage_outputs["video"] = video_path
                    return state, video_path, output, gr.update(visible=True), gr.update(visible=False)
                else:
                    return state, None, output, gr.update(visible=False), gr.update(visible=True)
            except Exception as e:
                state.logs = str(e)
                return state, None, str(e), gr.update(visible=False), gr.update(visible=True)
        
        def run_next_stage(stage_num, state):
            if not state.ssh_client or not state.ssh_client.get_transport() or not state.ssh_client.get_transport().is_active():
                return state, None, None, "", gr.update(visible=False), gr.update(visible=True)
            
            stage_map = {1: "tracking", 2: "trajectory", 3: "simulation", 4: "visualization"}
            stage = stage_map.get(stage_num)
            
            if not stage:
                return state, None, None, "", gr.update(visible=False), gr.update(visible=False)
            
            try:
                success, output = run_pipeline_stage(
                    stage, state.ssh_client, state.ssh_config,
                    state.output_dir_name, state.n_drones, state.user_prompt,
                    state.mount_path
                )
                
                state.logs = output
                state.current_stage = stage_num + 1
                
                if success:
                    base = os.path.join(state.mount_path, f"{state.output_dir_name}_{state.n_drones}_drones")
                    if stage == "tracking":
                        state.stage_outputs["tracking_video"] = os.path.join(base, "tracking", "tracking_visualization.mp4")
                        state.stage_outputs["points_only"] = os.path.join(base, "tracking", "tracking_points_only.mp4")
                        return state, state.stage_outputs["tracking_video"], state.stage_outputs["points_only"], output, gr.update(visible=True), gr.update(visible=False)
                    elif stage == "visualization":
                        state.stage_outputs["assignment"] = os.path.join(base, "simulation", "assignment.mp4")
                        state.stage_outputs["tracking_overlay"] = os.path.join(base, "simulation", "tracking_overlay.mp4")
                        return state, None, None, output, state.stage_outputs["assignment"], state.stage_outputs["tracking_overlay"]
                    return state, None, None, output, gr.update(visible=True), gr.update(visible=False)
                else:
                    return state, None, None, output, gr.update(visible=False), gr.update(visible=True)
            except Exception as e:
                state.logs = str(e)
                return state, None, None, str(e), gr.update(visible=False), gr.update(visible=True)
        
        def reset_all():
            return PipelineState()
        
        connect_btn.click(
            fn=connect_ssh,
            inputs=[show_name, user_prompt, n_drones, ssh_host, ssh_port, ssh_username, ssh_password, pipeline_state],
            outputs=[pipeline_state, connect_btn, start_btn]
        )
        start_btn.click(
            fn=run_video_stage,
            inputs=[pipeline_state],
            outputs=[pipeline_state, video_output, console1, continue1, retry1]
        )
        
        continue1.click(
            fn=lambda state: run_next_stage(1, state),
            inputs=[pipeline_state],
            outputs=[pipeline_state, tracking_video, points_only_video, console2, continue2, retry2]
        )
        
        reset_btn.click(
            fn=reset_all,
            outputs=[pipeline_state]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_port=7860, share=False)