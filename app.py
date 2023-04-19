from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
)
import gradio as gr
import torch
from PIL import Image
import time
import psutil
import random
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


start_time = time.time()
current_steps = 25

SAFETY_CHECKER = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker", torch_dtype=torch.float16)

UPSCALER = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16)
UPSCALER.to("cuda")
UPSCALER.enable_xformers_memory_efficient_attention()


class Model:
    def __init__(self, name, path=""):
        self.name = name
        self.path = path

        if path != "":
            self.pipe_t2i = StableDiffusionPipeline.from_pretrained(
                path, torch_dtype=torch.float16, safety_checker=SAFETY_CHECKER
            )
            self.pipe_t2i.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe_t2i.scheduler.config
            )
        else:
            self.pipe_t2i = None


models = [
    #Model("Stable Diffusion v1-4", "CompVis/stable-diffusion-v1-4"),
    # Model("Stable Diffusion v1-5", "runwayml/stable-diffusion-v1-5"),
    Model("anything-v4.0", "andite/anything-v4.0"),
]

MODELS = {m.name: m for m in models}

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"


def error_str(error, title="Error"):
    return (
        f"""#### {title}
            {error}"""
        if error
        else ""
    )


def inference(
    prompt,
    neg_prompt,
    guidance,
    steps,
    seed,
    model_name,
):

    print(psutil.virtual_memory())  # print memory usage

    if seed == 0:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator("cuda").manual_seed(seed)

    try:
        low_res_image, up_res_image = txt_to_img(
            model_name,
            prompt,
            neg_prompt,
            guidance,
            steps,
            generator,
        )
        return low_res_image, up_res_image, f"Done. Seed: {seed}",
    except Exception as e:
        return None, None, error_str(e)


def txt_to_img(
    model_name,
    prompt,
    neg_prompt,
    guidance,
    steps,
    generator,
):
    pipe = MODELS[model_name].pipe_t2i

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

    low_res_latents = pipe(
        prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        generator=generator,
        output_type="latent",
    ).images

    with torch.no_grad():
        low_res_image = pipe.decode_latents(low_res_latents)
        low_res_image = pipe.numpy_to_pil(low_res_image)

    up_res_image = UPSCALER(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=low_res_latents,
        num_inference_steps=20,
        guidance_scale=0,
        generator=generator,
    ).images

    pipe.to("cpu")
    torch.cuda.empty_cache()

    return low_res_image[0], up_res_image[0]


def replace_nsfw_images(results):
    for i in range(len(results.images)):
        if results.nsfw_content_detected[i]:
            results.images[i] = Image.open("nsfw.png")
    return results.images


with gr.Blocks(css="style.css") as demo:
    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div style="text-align: center">
                <h1>Anything v4 model + <a href="https://huggingface.co/stabilityai/sd-x2-latent-upscaler">Stable Diffusion Latent Upscaler</a></h1>
                <p>
                   Demo for the <a href="https://huggingface.co/andite/anything-v4.0">Anything v4</a> model hooked with the ultra-fast <a href="https://huggingface.co/stabilityai/sd-x2-latent-upscaler">Latent Upscaler</a>
                  </p>
              </div>
              <!-- 
              <p>To skip the queue, you can duplicate this Space<br>
              <a style="display:inline-block" href="https://huggingface.co/spaces/patrickvonplaten/finetuned_diffusion?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></p>
              -->
            </div>
        """
    )

    with gr.Column(scale=100):
        with gr.Group(visible=False):
                model_name = gr.Dropdown(
                    label="Model",
                    choices=[m.name for m in models],
                    value=models[0].name,
                    visible=False
                )

        with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                    with gr.Column():
                        prompt = gr.Textbox(
                            label="Enter your prompt",
                            show_label=False,
                            max_lines=1,
                            placeholder="Enter your prompt",
                            elem_id="prompt-text-input",
                        ).style(
                            border=(True, False, True, True),
                            rounded=(True, False, False, True),
                            container=False,
                        )
                        neg_prompt = gr.Textbox(
                            label="Enter your negative prompt",
                            show_label=False,
                            max_lines=1,
                            placeholder="Enter a negative prompt",
                            elem_id="negative-prompt-text-input",
                        ).style(
                            border=(True, False, True, True),
                            rounded=(True, False, False, True),
                            container=False,
                        )
                    generate = gr.Button("Generate image").style(
                        margin=False,
                        rounded=(False, True, True, False),
                        full_width=False,
                    )
            
        with gr.Accordion("Advanced Options", open=False):
                with gr.Group():
                    with gr.Row():
                        guidance = gr.Slider(
                            label="Guidance scale", value=7.5, maximum=15
                        )
                        steps = gr.Slider(
                            label="Steps",
                            value=current_steps,
                            minimum=2,
                            maximum=75,
                            step=1,
                        )

                    seed = gr.Slider(
                        0, 2147483647, label="Seed (0 = random)", value=0, step=1
                    )
            

    with gr.Column(scale=100):
        with gr.Row():
            with gr.Column(scale=75):
                up_res_image = gr.Image(label="Upscaled 1024px Image", shape=(1024, 1024))
            with gr.Column(scale=25):
                low_res_image = gr.Image(label="Original 512px Image", shape=(512, 512))
        error_output = gr.Markdown()

    inputs = [
        prompt,
        neg_prompt,
        guidance,
        steps,
        seed,
        model_name,
    ]
    outputs = [low_res_image, up_res_image, error_output]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

    ex = gr.Examples(
        [
            ["a mecha robot in a favela", "low quality", 7.5, 25, 33, models[0].name],
            ["the spirit of a tamagotchi wandering in the city of Paris", "low quality, bad render", 7.5, 50, 85, models[0].name],
        ],
        inputs=[prompt, neg_prompt, guidance, steps, seed, model_name],
        outputs=outputs,
        fn=inference,
        cache_examples=True,
    )
    ex.dataset.headers = [""]
    
    gr.HTML(
        """
    <div style="border-top: 1px solid #303030;">
      <br>
      <p>Space by 🤗 Hugging Face, models by Stability AI, andite, linaqruf and others ❤️</p>
      <p>This space uses the <a href="https://github.com/LuChengTHU/dpm-solver">DPM-Solver++</a> sampler by <a href="https://arxiv.org/abs/2206.00927">Cheng Lu, et al.</a>.</p>
      <p>This is a Demo Space For:<br>
      <a href="https://huggingface.co/stabilityai/sd-x2-latent-upscaler">Stability AI's Latent Upscaler</a>
    </div>
    """
    )

print(f"Space built in {time.time() - start_time:.2f} seconds")

demo.queue(concurrency_count=1)
demo.launch()
