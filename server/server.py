from flask import Flask, send_file
import jsonpickle
import numpy as np
from PIL import Image, ImageEnhance
from io import StringIO
import sys
from torchvision import transforms
import os

sys.path.insert(1, './model/')

from model import VariationalAutoencoder, torch

model = VariationalAutoencoder(36)

model.load_state_dict(torch.load('./model/VAE_1_0_0_CPU.pth'))
model.eval()

app = Flask(__name__)
@app.route('/', methods=['GET'])
def test():
    img = transforms.ToPILImage()(model.decoder(4*torch.rand((1,36)))[0]).resize((1024, 1024))

    enc = ImageEnhance.Sharpness(img)

    img = enc.enhance(2.5)

    img.save('./server/tmp.png')

    response = send_file('tmp.png')

    os.remove('./server/tmp.png')

    return response


# start flask app
app.run(host="0.0.0.0", port=8080)