import gradio as gr
import transformers
import numpy as np
import torch
from skimage import io
from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting
import torch

pipe_seg = transformers.pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
pipe_fundo = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe_inpainting = AutoPipelineForInpainting.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16)
pipe_inpainting.enable_model_cpu_offload()

from PIL import Image
import cv2

def generate_background(foreground_image, prompt_fundo):
    return pipe_fundo(prompt=prompt_fundo, num_inference_steps=1, guidance_scale=0.0, height=foreground_image.size[1], width=foreground_image.size[0]).images[0]

def remove_background(img_path, return_mask=False):
    return pipe_seg(img_path, return_mask=return_mask)

def replace_background(foreground_image, background_image, foreground_mask):
    foreground_pixels = foreground_image.load()
    background_pixels = background_image.load()
    mask_pixels = foreground_mask.load()
    for y in range(foreground_mask.size[1]):
        for x in range(foreground_mask.size[0]):
            if mask_pixels[x, y] == 0:
                foreground_pixels[x, y] = background_pixels[x, y]

    return foreground_image

def detect_faces(foreground_image, foreground_mask):
    foreground_image_np = np.array(foreground_image)
    new_mask = np.array(foreground_mask.copy())
    gray = cv2.cvtColor(foreground_image_np, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, minSize=(100, 100))
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        axes = (int(w * 0.5), int(h * 0.6))
        angle = 0
        color = (0, 0, 0)
        thickness = -1
        cv2.ellipse(new_mask, center, axes, angle, 0, 360, color, thickness)

    return Image.fromarray(new_mask)

def change_clothes(clothes_prompt, image_with_new_background, mask_with_face_rec):
    return pipe_inpainting(prompt=clothes_prompt, image=image_with_new_background, mask_image=mask_with_face_rec).images[0]

def run_pipeline(img_path, prompt_background, clothes_prompt):
    foreground_image = remove_background(img_path, return_mask=False)
    new_width = foreground_image.size[0] - (foreground_image.size[0] % 8)
    new_height = foreground_image.size[1] - (foreground_image.size[1] % 8)
    foreground_image = foreground_image.crop((0, 0, new_width, new_height))
    foreground_mask = remove_background(img_path, return_mask=True).crop((0, 0, new_width, new_height))
    background_image = generate_background(foreground_image, prompt_background)
    mask_with_face_rec = detect_faces(foreground_image, foreground_mask)
    image_with_new_background = replace_background(foreground_image, background_image,foreground_mask)


    return change_clothes(clothes_prompt, image_with_new_background, mask_with_face_rec)

with gr.Blocks() as demo:
    gr.Markdown("Pipeline para Finetuning automático do DreamBooth")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    fundo = gr.Textbox(label="Digite o prompt para o fundo", value="A photo of an cat", placeholder="Digite aqui o prompt")
                with gr.Column():
                    roupas = gr.Textbox(label="Digite o prompt para as roupas", placeholder="Digite aqui o prompt")
            imagem = gr.Image(type="filepath")
            btn = gr.Button(value="Trocar Fundo/roupas")
        with gr.Column():
            output_image = gr.Image(interactive=False)
            btn.click(run_pipeline, inputs=[imagem, fundo], outputs=output_image)


demo.launch(debug=True, share=True)