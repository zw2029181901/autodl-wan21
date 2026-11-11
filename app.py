import functools
from runtime.common import guarded, model_path, output_path, page_header, require_gpu, seed_generator


@functools.lru_cache(maxsize=1)
def pipeline():
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    require_gpu()
    directory = model_path()
    vae = AutoencoderKLWan.from_pretrained(directory, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(directory, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


@guarded
def generate(prompt, negative, frames, steps, seed):
    from diffusers.utils import export_to_video
    if not prompt.strip():
        raise ValueError("请输入视频内容")
    frames = int(frames)
    if frames not in (17, 33, 49, 65, 81):
        raise ValueError("请选择支持的帧数")
    video = pipeline()(prompt=prompt, negative_prompt=negative, height=480, width=832,
                       num_frames=frames, num_inference_steps=int(steps), guidance_scale=5.0,
                       generator=seed_generator(seed)).frames[0]
    destination = output_path(".mp4")
    export_to_video(video, str(destination), fps=16)
    return str(destination)


def build_ui():
    import gradio as gr
    with gr.Blocks(title="Wan2.1 文生视频") as ui:
        page_header()
        gr.Markdown("输出 832×480、16fps、无声视频。首次验收请选择 17 帧。")
        prompt = gr.Textbox(label="视频描述（建议英文）", value="A small boat glides across a calm lake at sunrise, cinematic, slow camera movement", lines=3)
        negative = gr.Textbox(label="负面描述", value="blurry, static, distorted, low quality")
        frames = gr.Dropdown([17, 33, 49, 65, 81], value=17, label="帧数")
        steps = gr.Slider(10, 40, value=25, step=1, label="步数")
        seed = gr.Number(value=42, precision=0, label="随机种子")
        button = gr.Button("生成短视频", variant="primary")
        result = gr.Video(label="视频结果")
        button.click(generate, [prompt, negative, frames, steps, seed], result, api_name="generate")
    return ui
