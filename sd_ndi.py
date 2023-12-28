import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

import time
import numpy as np
import cv2 as cv
import NDIlib as ndi

from pythonosc import udp_client
from pythonosc import osc_server
from pythonosc.dispatcher import Dispatcher
from threading import Thread
from typing import List, Any, Tuple
import json

def process_image(image_np: np.ndarray, range: Tuple[int, int] = (-1, 1)) -> Tuple[torch.Tensor, np.ndarray]:
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min
    return image.unsqueeze(0), image_np


def np2tensor(image_np: np.ndarray) -> torch.Tensor:
    height, width, _ = image_np.shape
    imgs = []
    img, _ = process_image(image_np)
    imgs.append(img)
    imgs = torch.vstack(imgs)
    images = torch.nn.functional.interpolate(
        imgs, size=(height, width), mode="bilinear", align_corners=False
    )
    image_tensors = images.to(torch.float16)
    return image_tensors

def prompt(address: str, *args: List[Any]) -> None:
    if address == "/prompt":
        global shared_message
        shared_message = args[0]

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

# Load config
config_data = load_config('config.json')
sd_model = config_data['sd_model']
t_index_list = config_data['t_index_list']
engine = config_data['engine']
min_batch_size = config_data['min_batch_size']
max_batch_size = config_data['max_batch_size']
osc_out_adress = config_data['osc_out_adress']
osc_out_port = config_data['osc_out_port']
osc_in_adress = config_data['osc_in_adress']
osc_in_port = config_data['osc_in_port']
print(config_data)

# You can load any models using diffuser's StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(sd_model).to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)

frame_buffer_size = 1

# Wrap the pipeline in StreamDiffusion
stream = StreamDiffusion(
    pipe,
    t_index_list=t_index_list,
    torch_dtype=torch.float16,
    frame_buffer_size = frame_buffer_size
)

# If the loaded model is not LCM, merge LCM
stream.load_lcm_lora()
stream.fuse_lora()
# Use Tiny VAE for further acceleration
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
# Enable acceleration
stream = accelerate_with_tensorrt(
    stream, engine, min_batch_size=min_batch_size ,max_batch_size=max_batch_size
)

prompt = "banana in space"
# Prepare the stream
stream.prepare(prompt)

# NDI
ndi_find = ndi.find_create_v2()

sources = []
while not len(sources) > 0:
    print('Looking for sources ...')
    ndi.find_wait_for_sources(ndi_find, 1000)
    sources = ndi.find_get_current_sources(ndi_find)

print('NDI connected')

ndi_recv_create = ndi.RecvCreateV3()
ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
ndi_recv = ndi.recv_create_v3(ndi_recv_create)
ndi.recv_connect(ndi_recv, sources[0])
ndi.find_destroy(ndi_find)
cv.startWindowThread()
send_settings = ndi.SendCreate()
send_settings.ndi_name = 'SD-NDI'
ndi_send = ndi.send_create(send_settings)
video_frame = ndi.VideoFrameV2()

# OSC
server_address = osc_out_adress
server_port = osc_out_port
client = udp_client.SimpleUDPClient(server_address, server_port)

server_address = osc_in_adress
server_port = osc_in_port
shared_message = None

dispatcher = Dispatcher()
dispatcher.map("/prompt", prompt)

server = osc_server.ThreadingOSCUDPServer(
      (server_address, server_port), dispatcher)

server_thread = Thread(target=server.serve_forever)
server_thread.start()

# Run the stream infinitely
try:
    while True:
        if shared_message is not None:

            prompt = str(shared_message)
            stream.prepare(prompt)
            # Process the received message within the loop as needed
            print(f"Prompt: {prompt}")
            # Reset the shared_message variable
            shared_message = None

        t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 5000)

        if t == ndi.FRAME_TYPE_VIDEO:

            frame = np.copy(v.data)
            framergb = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

            inputs = []

            inputs.append(np2tensor(framergb))

            if len(inputs) < frame_buffer_size:
                time.sleep(0.005)
                continue
            start_time = time.time()
            sampled_inputs = []
            for i in range(frame_buffer_size):
                index = (len(inputs) // frame_buffer_size) * i
                sampled_inputs.append(inputs[len(inputs) - index - 1])
            input_batch = torch.cat(sampled_inputs)
            inputs.clear()
            output_images = stream(
                input_batch.to(device=stream.device, dtype=stream.dtype)
            ).cpu()
            if frame_buffer_size == 1:
                output_images = [output_images]
            for output_image in output_images:
                output_image = postprocess_image(output_image, output_type="np")[0]

                open_cv_image = (output_image * 255).round().astype("uint8")

                img = cv.cvtColor(open_cv_image, cv.COLOR_RGB2RGBA)
                ndi.recv_free_video_v2(ndi_recv, v)

                video_frame.data = img
                video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX

                ndi.send_send_video_v2(ndi_send, video_frame)

            fps = 1 / (time.time() - start_time)

            client.send_message("/fps", fps)

except KeyboardInterrupt:
    # Handle KeyboardInterrupt (Ctrl+C)
    print("KeyboardInterrupt: Stopping the server")
finally:
    # Stop the server when the loop exits
    ndi.recv_destroy(ndi_recv)
    ndi.send_destroy(ndi_send)
    ndi.destroy()
    cv.destroyAllWindows()
    server.shutdown()
    server_thread.join()
