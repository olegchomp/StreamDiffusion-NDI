# StreamDiffusion-NDI
<a href="https://discord.com/invite/wNW8xkEjrf"><img src="https://discord.com/api/guilds/838923088997122100/widget.png?style=shield" alt="Discord Shield"/></a>

NDI & OSC extension for real-time Stable Diffusion interactive generation with [StreamDiffusion](https://pages.github.com/](https://github.com/cumulo-autumn/StreamDiffusion)https://github.com/cumulo-autumn/StreamDiffusion).

## Features:
* NDI (video) streaming over the network or localhost.
* OSC communication for prompt and FPS.

## Installation:
Supported version Python 3.10
1) Install [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion?tab=readme-ov-file#installation) with TensorRT.
2) Install SteamDiffusion-NDI in StreamDiffusion environment with ```pip install -r requirements.txt```

## Usage:
1) Accelerate model with an [example](https://github.com/cumulo-autumn/StreamDiffusion/blob/main/examples/img2img/single.py) script  (temporal solution). Change [acceleration](https://github.com/cumulo-autumn/StreamDiffusion/blob/63a240a771247968b0fed9877d3c0436d3110b86/examples/img2img/single.py#L24C7-L24C7) to ```tensorrt```.
2) Configure ```config.json```
3) Run in StreamDiffusion environment ```python sd_ndi.py```
4) Add NDI output to send images in Stable Diffusion and NDI input (SD-NDI) to receive processed images
5) Send string with OSC at ```/prompt``` address change the prompt during the inference
6) You can get inference FPS at ```/fps``` address on client side

## Config:
Look in ```config.json``` for an example configuration.

	"sd_model": "path to diffusers model",
	"t_index_list": [Number of inference steps],
	"engine": "path to the folder with the accelerated model",
	"min_batch_size": depends on your configuration,
	"max_batch_size": depends on your configuration,
	"osc_out_adress": client address for receiving FPS value,
	"osc_out_port": client port for receiving FPS value,
	"osc_in_adress": server address for receiving commands.,
	"osc_in_port": server port for receiving commands.
