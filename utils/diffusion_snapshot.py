# import torch
# from diffusers import StableDiffusionImg2ImgPipeline
# from PIL import Image

# # Load once (global, cached)
# pipe = None

# def load_pipeline(device):
#     global pipe
#     if pipe is None:
#         pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#             torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
#             safety_checker=None
#         ).to(device)
#     return pipe


# def generate_snapshot(
#     input_image,
#     prompt,
#     strength=0.6,
#     guidance=7.5,
#     device=torch.device("cpu")
# ):
#     try:
#         pipe = load_pipeline(device)

#         result = pipe(
#             prompt=prompt,
#             image=input_image,
#             strength=strength,
#             guidance_scale=guidance,
#             num_inference_steps=25
#         )

#         if result is None or len(result.images) == 0:
#             print("Diffusion returned no images.")
#             return None

#         return result.images[0]

#     except Exception as e:
#         print("Diffusion error:", e)
#         return None


#     pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
#     safety_checker=None
# ).to(device)


import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = None

def load_pipeline(device):
    global pipe
    if pipe is None:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            safety_checker=None
        ).to(device)
    return pipe


def generate_snapshot(
    input_image,
    prompt,
    device,
    strength=0.40,
    guidance=7,
    steps=18
):
    try:
        pipe = load_pipeline(device)

        # 🔽 CRITICAL: downscale before diffusion
        input_image = input_image.resize((384, 384), Image.BICUBIC)

        with torch.no_grad():
            with torch.autocast(device.type if device.type == "cuda" else "cpu"):
                result = pipe(
                    prompt=prompt,
                    image=input_image,
                    strength=strength,
                    guidance_scale=guidance,
                    num_inference_steps=steps
                )

        if result is None or not result.images:
            return None

        return result.images[0]

    except Exception as e:
        print("Diffusion failed:", e)
        return None
