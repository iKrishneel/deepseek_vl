#!/usr/bin/env python

import io

from flask import Flask, request, jsonify
from PIL import Image

from interactive_inference import load_image, remote_chat


app = Flask(__name__)


model = remote_chat()

@app.route('/chat', methods=['POST'])
def upload_image():
    if 'image' in request.files:
        image_file = request.files['image']
        image = [load_image(image_file)]
    else:
         image = []

    description = request.form.get('text', '')
    prompt = f'<image_placeholder>{description}'

    answer = model(image, prompt)    
    return jsonify(
        {
            'response': {
                'description': description,
                'answer': answer,
            },
        }
    )


@app.route('/send_text', methods=['POST'])
def receive_text():
    data = request.json
    client_text = data.get('text', '')

    server_response = f"Server received: {client_text}"
    return jsonify({'response': server_response})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
