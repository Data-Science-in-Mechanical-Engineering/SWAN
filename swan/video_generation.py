import json
import uuid
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, TextStreamer
from typing import Dict, Any, List, Tuple
import multiprocessing as mp
import tempfile
import sys
import subprocess
import aiohttp
import asyncio
import os
import shutil
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class StructuredPrompt:
    original_prompt: str
    segmentation_prompt: str
    # video generation prompt elements
    cinematic_enhancement: str
    camera_angle_and_shot_type: str
    scene_description: str
    motion_description: str


@dataclass
class PromptExpansionConfig:
    model_path: str = "/weights/Qwen3.5-9B"
    use_8bit_quantization: bool = True
    device: str = "cuda"
    enable_thinking: bool = False
    max_new_tokens: int = 2**14
    system_prompt: str = open("/app/static_data/comfyui/system_prompt.md", "r").read()

@dataclass
class ComfyUIConfig:
    main_path: str = "/app/comfyui/main.py"
    output_dir: str = "/app/comfyui/output/" # only for intermediate files, i.e. generated start frames are copied here so ComfyUI can access them
    server_port: int = 8188
    server_url: str = "127.0.0.1"
    server_log_path: str = "/app/out/comfyui_server.log"
    extra_models_yaml_path: str = "/app/static_data/comfyui/extra_models.yaml"
    use_flash_attention: bool = True
    image_generation_workflow_name: str = "z_image_turbo"
    image_generation_workflow_path: str = "/app/static_data/comfyui/workflows/z_image_turbo.json"
    video_generation_workflow_name: str = "I2V_Wan_FP8_720p_step_distillation"
    video_generation_workflow_path: str = "/app/static_data/comfyui/workflows/I2V_Wan_FP8_720p_step_distillation.json"

@dataclass
class VideoGenerationConfig:
    # use default factories for the nested configs to avoid mutable default arguments issues
    prompt_expansion_config: PromptExpansionConfig = field(default_factory=PromptExpansionConfig)
    comfyui_config: ComfyUIConfig = field(default_factory=ComfyUIConfig)
    resolution: Tuple[int, int] = (1280, 720) # Default resolution for the wan model
    image_noise_seed: int | None = None
    video_noise_seed: int | None = None

class ComfyUIServer:
    """
    A class to manage the lifecycle of a ComfyUI server instance, allowing for programmatic startup and shutdown.
    """
    def __init__(self, comfyui_config: ComfyUIConfig):
        """Initialize the ComfyUIServer.
            Args:
                comfyui_config: An instance of ComfyUIConfig containing the configuration for the server, including paths, ports, and other settings.
        """
        self.comfyui_main_path = comfyui_config.main_path
        self.port = comfyui_config.server_port
        self.server_url = comfyui_config.server_url
        self.use_flash_attention = comfyui_config.use_flash_attention
        self.log_file = comfyui_config.server_log_path
        self.extra_models_yaml_path = comfyui_config.extra_models_yaml_path
        self.log_handle = None
        self.process = None

    async def start(self):
        """Manually start the ComfyUI server."""
        print(f'Starting ComfyUI server on port {self.port}...')
        self.log_handle = open(self.log_file, 'w')

        # print all arguments for debugging
        print(f"ComfyUI main path: {self.comfyui_main_path}")
        print(f"ComfyUI server log file: {self.log_file}")
        print(f"ComfyUI extra models yaml path: {self.extra_models_yaml_path}")
        print(f"ComfyUI server URL: {self.server_url}")
        print(f"ComfyUI use flash attention: {self.use_flash_attention}")
        print(f"ComfyUI command: {sys.executable} {self.comfyui_main_path} --listen {self.server_url} --port {self.port} {'--use-flash-attention' if self.use_flash_attention else ''} --extra-model-paths-config {self.extra_models_yaml_path}")

        self.process = await asyncio.create_subprocess_exec(
            sys.executable, self.comfyui_main_path, 
            '--listen', self.server_url,
            '--port', str(self.port),
            '--use-flash-attention' if self.use_flash_attention else '',
            '--extra-model-paths-config', self.extra_models_yaml_path,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT
        )
        await self._wait_for_server_startup()
        print('ComfyUI server started successfully.')

    async def stop(self):
        """Manually stop the ComfyUI server."""
        print('Shutting down ComfyUI server...')
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                print('ComfyUI server did not shut down in time. Killing process...')
                self.process.kill()
        if self.log_handle:
            self.log_handle.close()
        print('ComfyUI server stopped.') 
    
    async def _wait_for_server_startup(self, timeout=60):
        """Waits for the ComfyUI server to be ready to accept requests."""
        start_time = asyncio.get_event_loop().time()
        url = f'http://{self.server_url}:{self.port}/system_stats'

        if self.process is None:
            raise Exception('ComfyUI server process has not been started.')

        async with aiohttp.ClientSession() as session:
            while True:
                if self.process.returncode is not None:
                    raise Exception(f"ComfyUI server process terminated unexpectedly during startup. Exit code: {self.process.returncode}")
                if asyncio.get_event_loop().time() - start_time > timeout:
                    self.process.terminate()
                    raise Exception('Timed out waiting for ComfyUI server to start.')

                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            print('ComfyUI server is ready to accept requests.')
                            return
                except aiohttp.ClientError:
                    await asyncio.sleep(1)  # Wait a bit before retrying        

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

class ComfyUIClient:
    """
    Handles async communication with the ComfyUI server.
    """
    def __init__(self, comfyui_config: ComfyUIConfig):
        self.base_url = f'http://{comfyui_config.server_url}:{comfyui_config.server_port}'
        self.comfyui_output_dir = comfyui_config.output_dir
        # check if the output directory exists
        if not os.path.isdir(self.comfyui_output_dir):
            raise Exception(f"ComfyUI output directory does not exist: {self.comfyui_output_dir}")

    async def upload_image(self, session: aiohttp.ClientSession, image_path: str) -> str:
        """
        Upload an image to the ComfyUI server and returns the internal comfyui image filename.
        - Args:
            - `image_path`: The file path of the image to upload.
        - Returns:
            - The filename of the uploaded image as recognized by ComfyUI, which can be used as an input in the workflow JSON.
        """

        with open(image_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('image', f, filename=os.path.basename(image_path), content_type='image/png')
            data.add_field('overwrite', 'true')  # Add overwrite field to allow overwriting existing images with the same name

            async with session.post(f'{self.base_url}/upload/image', data=data) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Failed to upload image. Status {response.status}: {text}")

                result = await response.json()
                print(f"Uploaded image {image_path} to ComfyUI as {result['name']}")
                return result['name']
            
    async def execute_workflow(self, session: aiohttp.ClientSession, workflow: Dict[str, Any], output_map: Dict[str, str], poll_interval_seconds: float = 2.0) -> str:
        """
        Execute a workflow on the ComfyUI server, wait for it to complete and copy the generated assets to their target locations.
        - Args:
            - `workflow`: The workflow JSON as a dictionary, with any necessary parameters filled in.
            - `output_map`: A dictionary mapping generated asset UUIDs (as specified in the workflow JSON) to their target output file paths. This is used to move the generated files from the ComfyUI output directory to their target locations.
        """
        
        payload = {"prompt": workflow}
        async with session.post(f'{self.base_url}/prompt', json=payload) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Failed to queue workflow. Status {response.status}: {text}")
            data = await response.json()
            prompt_id = data["prompt_id"]

        print(f"Workflow queued with prompt ID: {prompt_id}. Waiting for completion...")
        while True:
            async with session.get(f"{self.base_url}/history/{prompt_id}") as response:
                if response.status == 200:
                    history_data = await response.json()
                    if prompt_id in history_data:
                        print(f"Workflow with prompt ID {prompt_id} completed. Retrieving assets...")
                        self._retrieve_outputs(output_map)
                        return prompt_id
            await asyncio.sleep(poll_interval_seconds)  # Wait before polling again
        
    def _retrieve_outputs(self, output_map: Dict[str, str]):
        """Move generated files from the ComfyUI output directory to their target locations as specified in the output map."""
        
        for file_uuid, target_path in output_map.items():
            matching_files = list(Path(self.comfyui_output_dir).rglob(f"*{file_uuid}*"))
            if not matching_files:
                print(f"No files found for UUID: {file_uuid}")
                continue
            elif len(matching_files) > 1:
                print(f"Multiple files found for UUID {file_uuid}. Using the first match: {matching_files[0]}")
            
            source_file = matching_files[0]
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.move(str(source_file), target_path)
            print(f"Moved generated file from {source_file} to {target_path}")
        
        # run the same loop again to actually raise an error if any files were not found, instead of silently failing
        for file_uuid, target_path in output_map.items():
            if not os.path.isfile(target_path):
                raise Exception(f"Expected output file not found for UUID {file_uuid} at target path {target_path}. Check if the workflow generated the expected output and if the output map is correct.") 


# The LLM leaks memory, so we run the expansion in a separate process that can be cleanly killed after the expansion is done. 
class PromptExpander:
    def __init__(self, p_config: PromptExpansionConfig):
        self.p_config = p_config
        
        self.processor = AutoProcessor.from_pretrained(self.p_config.model_path, trust_remote_code=True)
        self.streamer = TextStreamer(self.processor.tokenizer)
        if self.p_config.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.p_config.model_path,
                quantization_config=bnb_config,
                device_map=self.p_config.device,
                attn_implementation="flash_attention_2", # FA2 is still fully supported
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.p_config.model_path,
                device_map=self.p_config.device,
                attn_implementation="flash_attention_2",
            )

    def expand(self, system_prompt: str, user_prompt: str) -> Tuple[str, str]:
        """
        Processes the input prompts and generates an expanded prompt using the model.
        - Args:
            - `system_prompt`: A string providing instructions or context for the model.
            - `user_prompt`: The original prompt that needs to be expanded.
        - Returns:
            - An expanded version of the user prompt generated by the model.
            - The 'thought' process as a string.
        """
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]

        with torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                enable_thinking=self.p_config.enable_thinking,
                return_tensors="pt",
            ).to(self.p_config.device)

            outputs = self.model.generate(**inputs, max_new_tokens=self.p_config.max_new_tokens, streamer=None)

            generated_text = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            )
            del inputs, outputs
        
        # The message starts with the thoughts block '</think>'
        if self.p_config.enable_thinking:
            if "</think>" in generated_text:
                parts = generated_text.split("</think>") # split the output into the part before and after </think>
                thoughts_part = parts[0].strip() # the part before </think> should contain the thoughts
                answer_part = parts[1].strip() # the part after </think> should contain
            else:
                thoughts_part = generated_text.strip()
                answer_part = "Error: Generation cut off before </think> tag. Increase max_new tokens"
        else:
                answer_part = generated_text.strip()
                thoughts_part = ""
        return answer_part, thoughts_part


def _prompt_expansion_worker(p_config: PromptExpansionConfig, system_prompt: str, user_prompt: str):
    expander = PromptExpander(p_config)
    return expander.expand(system_prompt, user_prompt)

def expand_prompt(user_prompt: str, p_config: PromptExpansionConfig) -> StructuredPrompt:
    """
    Convenience wrapper around expand_prompts for a single prompt.
    - Args:
        - `user_prompt`: The original prompt that needs to be expanded.
        - `p_config`: The configuration for the prompt expansion process.
    - Returns:
        - A `StructuredPrompt` containing the components for video generation and further processing.
    """

    # Run the prompt expansion in a separate process to avoid memory leaks in the LLM
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        result = pool.apply(
            _prompt_expansion_worker, 
            args=(p_config, p_config.system_prompt, user_prompt)
        )
    answer, thoughts = result


    # answer should be a JSON object with keys "Cinematic Enhancement", "Camera Angle and Shot Type", "Scene Description", "Motion Description", "Segmentation Prompt"
    try:
        answer_dict = json.loads(answer)
        structured_prompt = StructuredPrompt(
            original_prompt=user_prompt,
            segmentation_prompt=answer_dict["Segmentation Prompt"],
            cinematic_enhancement=answer_dict["Cinematic Enhancement"],
            camera_angle_and_shot_type=answer_dict["Camera Angle and Shot Type"],
            scene_description=answer_dict["Scene Description"],
            motion_description=answer_dict["Motion Description"],
        )
    except Exception as e:
        print(f"Error parsing answer: {e}")
        structured_prompt = StructuredPrompt(
            original_prompt=user_prompt,
            segmentation_prompt="Error parsing segmentation prompt",
            cinematic_enhancement="Error parsing cinematic enhancement",
            camera_angle_and_shot_type="Error parsing camera angle and shot type",
            scene_description="Error parsing scene description",
            motion_description="Error parsing motion description",
        )

    return structured_prompt


class WorkflowTemplate:
    """
    Creates templates to easily fill comfyui workflows with parameters.
    The presets here each match a specific workflow template JSON file in /app/static_data/comfyui/workflows/
    Important: output paths must begin with "output_". This is because ComfyUI generates the output file names dynamically with an incrementing number suffix.
    We manually copy the generated assets from the ComfyUI output directory to the specified output path. 
    """
    PRESETS = {
        "z_image_turbo": {
            "prompt": ("45", "text"),
            "noise_seed": ("44", "seed"),
            "width": ("41", "width"),
            "height": ("41", "height"),
            "output_image": ("9", "filename_prefix")
        },
        "I2V_Wan_FP16_720p_no_distillation": {
            "prompt": ("6", "text"),
            "noise_seed": ("57", "noise_seed"),
            "input_image": ("62", "image"),
            "output_video": ("61", "filename_prefix"),
        },
        "I2V_Wan_FP8_720p_step_distillation": {
            "prompt": ("93", "text"),
            "noise_seed": ("86", "noise_seed"),
            "input_image": ("97", "image"),
            "output_video": ("108", "filename_prefix"),
        },
        "I2V_Hunyuan_FP16_720p_no_distillation": {
            "prompt": ("44", "text"),
            "noise_seed": ("127", "noise_seed"),
            "input_image": ("80", "image"),
            "output_video": ("102", "filename_prefix"),
        },
        "I2V_Hunyuan_FP8_720p_cfg_distillation": {
            "prompt": ("44", "text"),
            "noise_seed": ("127", "noise_seed"),
            "input_image": ("80", "image"),
            "output_video": ("102", "filename_prefix"),
        },
        "T2V_Wan_FP16_720p_no_distillation": {
            "prompt": ("99", "text"),
            "noise_seed": ("96", "noise_seed"),
            "output_video": ("98", "filename_prefix"),
        },
        "T2V_Wan_FP8_720p_step_distillation": {
            "prompt": ("89", "text"),
            "noise_seed": ("81", "noise_seed"),
            "output_video": ("80", "filename_prefix"),
        },
        "T2V_Hunyuan_FP16_720p_no_distillation": {
            "prompt": ("44", "text"),
            "noise_seed": ("129", "noise_seed"),
            "output_video": ("102", "filename_prefix"),
        },
    }

    def __init__(self, template_path, preset_name: str):
        self.template_path = template_path
        self.preset_name = preset_name
        
        if preset_name not in self.PRESETS:
            raise ValueError(f"Preset '{preset_name}' not found. Available presets: {list(self.PRESETS.keys())}")
        self.template = self.PRESETS[preset_name]

        with open(template_path, 'r') as f:
            self.workflow = json.load(f)

    def build(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Creates the workflow json from the template and fill in the all given parameters. 
        Raises ComfyUIError if any required parameter is missing. Ignores any extra parameters.
            Args:
                **kwargs: The parameters to fill in the workflow template. The keys should match the argument names defined in the preset.
            Returns:
                workflow: A dictionary representing the comfyui workflow JSON with the filled parameters. 
                output_map: A dictionary mapping generated asset UUIDs to their specified output paths.
        """
        import copy
        workflow = copy.deepcopy(self.workflow)

        # Maps comyUI UUID prefixes to absolute output file paths we specify
        output_map = {}

        for arg_name, arg_value in kwargs.items():
            if arg_name not in self.template:
                print(f"Argument '{arg_name}' is not in the template and will be ignored.")
                continue
                
            node_id, input_key = self.template[arg_name]

        # Each output asset receives a UUID, later we can use output map to move all generated assets to their target locations.
        # Necessary because ComfyUI generates the output file names dynamically with an incrementing number suffix.
            if arg_name.startswith("output_"):
                asset_uuid = str(uuid.uuid4())
                workflow[node_id]['inputs'][input_key] = asset_uuid
                output_map[asset_uuid] = arg_value
            else:
                workflow[node_id]['inputs'][input_key] = arg_value
        return workflow, output_map

async def generate_video(config: VideoGenerationConfig, user_prompt: str, image_output_path: str, video_output_path: str, prompt_output_path: str):
    
    print("Expanding prompt...")
    structured_prompt = expand_prompt(user_prompt, config.prompt_expansion_config)
    print("Structured Prompt:")
    print(structured_prompt)
    
    with open(prompt_output_path, 'w') as f:
        json.dump(structured_prompt.__dict__, f, indent=4)

    # image prompt
    image_prompt = structured_prompt.cinematic_enhancement + "." + structured_prompt.camera_angle_and_shot_type + "." + structured_prompt.scene_description + "."
    image_prompt = image_prompt.replace("..", ".") # remove any double dots that may occur if the sections already end with a dot
    video_prompt = image_prompt + ". " + structured_prompt.motion_description
    video_prompt = video_prompt.replace("..", ".")
    
    image_generation_workflow_template = WorkflowTemplate(config.comfyui_config.image_generation_workflow_path, config.comfyui_config.image_generation_workflow_name)
    image_noise_seed = config.image_noise_seed if config.image_noise_seed is not None else np.random.randint(0, 1000000)
    video_generation_workflow_template = WorkflowTemplate(config.comfyui_config.video_generation_workflow_path, config.comfyui_config.video_generation_workflow_name)
    video_noise_seed = config.video_noise_seed if config.video_noise_seed is not None else np.random.randint(0, 1000000)

    image_workflow, image_output_map = image_generation_workflow_template.build(
        prompt=image_prompt,
        output_image=image_output_path,
        noise_seed=image_noise_seed,
        width=config.resolution[0],
        height=config.resolution[1],
    )

    server = ComfyUIServer(comfyui_config=config.comfyui_config)
    await server.start() # start comfyui server in separate process
    client = ComfyUIClient(comfyui_config=config.comfyui_config)
    async with aiohttp.ClientSession() as session:
        print("Generating start frame...")
        await client.execute_workflow(session=session, workflow=image_workflow, output_map=image_output_map)
        input_image_path = await client.upload_image(session=session, image_path=image_output_path)

        video_workflow, video_output_map = video_generation_workflow_template.build(
            prompt=video_prompt,
            input_image=input_image_path, # has to be inside the comfyui output directory to be accessible by comfyui.
            output_video=video_output_path,
            noise_seed=video_noise_seed,
        )
        print("Generating video...")
        await client.execute_workflow(session=session, workflow=video_workflow, output_map=video_output_map)