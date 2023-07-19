from flask import Flask, send_file
import jsonpickle
import numpy as np
from PIL import Image, ImageEnhance
from io import StringIO
import sys
from torchvision import transforms
import os
import argparse
sys.path.insert(1, './model/')
from model import VariationalAutoencoder, torch

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', type=str)
    parser.add_argument('--port', type=int, default=8080)

    args = parser.parse_args()

    model = VariationalAutoencoder(36)

    model.load_state_dict(torch.load(args.filepath))
    model.eval()

    app.run(host="0.0.0.0", port=args.port)