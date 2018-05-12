# 6.883VAEDemo
VAE Demo Page for MIT Spring 2018 6.883

## Content
```
.
├── index.html
├── model
│   ├── decoder_generator.png
│   ├── encoder.py
│   ├── mnist-cnn-vae.py
│   ├── model.bin
│   ├── model.h5
│   └── model_pb2.py
└── README.md
```
 - `index.html`: all html, javascript, css in one file
 - `minst-cnn-vae.py`: define and train CNN-VAE model, run with `python3 minst-cnn-vae.py`, will produce `decoder_generator.png` and `model.h5`
 - `decoder_generator.png`: a visualization of the generated images
 - `encoder.py`: (which use `model_pb2.py`): weight-wise quantization to 8-bit float, convert `model.h5` to `model.bin`, which is used by the javascript in `index.html`
 
 ## Replacing the Model
 - Train a Keras model and save as `model.h5`
 - Convert the model by `encoder.py`, get `model.bin`, place in `./model/` folder
 - Change the name of input layer and output layer [line 43](https://github.com/johnding1996/6.883VAEDemo/blob/da0c69e76e7bca7083e36b2855d8b07076392ccc/index.html#L43) and [line 45](https://github.com/johnding1996/6.883VAEDemo/blob/da0c69e76e7bca7083e36b2855d8b07076392ccc/index.html#L45) in `index.html`
 - Check it works
  
## Improve the UI
 - If you have any ideas and suggestions on improving the UI, please create a issue!
