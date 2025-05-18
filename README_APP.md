

```bash


docker run --gpus device=2 --shm-size=32g -it --net host -v ./pytorch-CycleGAN-and-pix2pix/:/pytorch-CycleGAN-and-pix2pix/ -v ./unpaired_self_datasets/:/data kevinchina/deeplearning:2.5.1-cuda12.1-cudnn9-devel-pix2pix-webui bash

python colorization_app.py
```

