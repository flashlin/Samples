
|Model Size |Minimum Run Size Requirement
|--|--
|4bit-7B |6GB+ 
|4bit-13B |10GB+ 
|4bit-30B |20GB+ 
|4bit-65B |40GB+ 

The above is the model execution requirements size. If you want to train and fine-tune, you must multiply by 2.

|CPU |RAM |GPU VRAM Size |DISK |Limit & Reason | Price
|--|--|--|--|--|--
|i9-13900K |ECC-128GB |NVIDIA-A100 80GB PCIE|2TB SSD |Capable of meeting the operational needs of most models. |x
|i9-13900K |ECC-128GB |NVIDIA-RTX-A6000 48GB PCIE|2TB SSD |You cannot simultaneously run the 65B model on GPU and develop. |413,935
|i9-13900K |ECC-128GB |NVIDIA-RTX-4090 24GB PCIE x 2|2TB SSD |The maximum model that can be executed is 30B | 343,952

292,624

Not all models support splitting the model across more than two GPUs. We should prioritize models with large VRAM capacity.

Although the industry has released many large models, they are often difficult to train and deploy. The 48GB size is just right in the middle. I believe that models will become smaller in the future.

Furthermore, we can utilize RAM to run the model at a slower pace for fine-tuning tasks, all while ensuring sufficient resources are allocated for the website's operation.

Considering the model size and usage, I recommend for this hardware setup.

|CPU |RAM |GPU VRAM Size |DISK 
|--|--|--|--
|i9-13900K |ECC-128GB |NVIDIA-RTX-A6000 48GB PCIE|2TB SSD 


For supporting Stable Diffusion XL for 1024x1024 image size, 32G is required. Hence, the recommended spec for Diffusion models is good enough.


|Model Name |Model Size
|--|--
|Diffusion Models |0.1B
|Big Diffusion Models |1B
|Diffusion Models with Attention |1.5B
|Stable Diffusion	| 2.5B
|Stable Diffusion XL |	4.6B

The following are the VRAM requirements of Stable Diffusion XL
|Image Size |Required vram size
|--|--
|64x64 | 2GB
|128x128 |4GB
|256x256 |8GB
|512x512 |16GB
|1024x1024 |32GB





<br><br><br><br><br><br><br><br><br>

### Why won’t LLM 13B fit on my 4090? OOM
|2x<br>RTX-4090 24GB|2x<br> RTX-A6000 48GB |2x<br>RTX-A100 80GB
|--|--|--
|$ for Mid-Size Models |More GPU Memory for Large Models |Max Performance with More Large Models
|16-Cores |32-Cores |64-Cores
|128 GB of system memory |256 GB of system memory |256 GB of system memory 


<br><br><br><br><br><br><br><br><br>



申請一台電腦工作站規格如下, 請核准
|種類|規格
|--|--
|CPU |i7-13700K
|RAM |ECC-96GB
|GPU |NVIDIA-RTX-A6000 48GB PCIE 
|DISK |2TB SSD
|OS |Ubuntu-22.04 