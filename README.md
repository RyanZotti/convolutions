# Convolutions and Transposed Convolutions

Perform convolutions and transposed convolutions, saving the output along the way.

## CLI Inputs

| Args | Description | Default |
|------|-------------|---------|
| --image | The path to the image file | images/python.png |
| --num-convolutions | The number of convolutions (and deconvolutions) to perform | 3 |

## How to run the script

```python
python convolutions.py --image <path to image> --num-convolutions <number of convolutions>
```

For example:

```python
python convolutions.py --image images/python.png --num-convolutions 3
```