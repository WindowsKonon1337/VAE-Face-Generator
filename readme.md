# VAE Face Generator

## I know that GAN is better than VAE for this task. This is just a practice challenge

## Installing libraries

```bash
    ./install_libs.sh
```

## Dataset: CelebA

available from torchvision.datasets

## Network weights: 

Model weights availabe [here](https://cloud.mail.ru/public/ephV/K6y6B2BZ3)

Drag .pth file in `model` folder

## Launch

```bash
    python3 ./server/server.py --filepath=<path to .pth file> --port=<port. Default: 8080>
```
