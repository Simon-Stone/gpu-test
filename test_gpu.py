import pytest

from diffusers import StableDiffusionPipeline
import numpy as np
from numba import cuda
from sentence_transformers import SentenceTransformer
import torch
import torchvision.models as models
import torchvision.transforms as T
import whisper
from PIL import Image

import os


DEVICES = ["cpu", "mps", "cuda"]


@pytest.mark.parametrize("device", DEVICES)
def test_conv2d_forward(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    x = torch.randn(1, 3, 224, 224, device=device)

    conv = torch.nn.Conv2d(
        in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
    ).to(device)

    y = conv(x)

    assert y.shape == (1, 16, 224, 224)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("device", DEVICES)
def test_numba_vec_add(device):
    if device != "cuda" or not torch.cuda.is_available():
        pytest.skip("Numba test only runs on CUDA")

    devices = cuda.gpus
    if len(devices) == 0:
        pytest.skip("Numba sees zero CUDA devices")

    numba_device = devices[0]

    N = 10_000_000
    a_host = np.random.rand(N).astype(np.float64)
    b_host = np.random.rand(N).astype(np.float64)

    a_dev = cuda.to_device(a_host, to=numba_device)
    b_dev = cuda.to_device(b_host, to=numba_device)
    out_dev = cuda.device_array_like(a_host)

    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    @cuda.jit
    def vec_add(dst, src1, src2):
        i = cuda.grid(1)
        if i < dst.size:
            dst[i] = src1[i] + src2[i]

    # Warm‑up launch
    vec_add[blocks_per_grid, threads_per_block](out_dev, a_dev, b_dev)
    cuda.synchronize()

    # Real launch
    vec_add[blocks_per_grid, threads_per_block](out_dev, a_dev, b_dev)
    cuda.synchronize()

    out_host = out_dev.copy_to_host()

    assert out_host.shape == (N,)
    assert np.isfinite(out_host).all()


@pytest.mark.parametrize("device", DEVICES)
def test_resnet18_forward(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    model.eval()

    img = Image.open("./dog.jpg").convert("RGB")
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 1000)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("device", DEVICES)
def test_sentence_transformer(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device,
    )

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Testing GPU inference pipelines.",
        "OpenAI's models are fun to play with.",
    ]

    embeddings = model.encode(sentences, show_progress_bar=False)
    assert len(embeddings) == 3
    assert embeddings[0].shape[0] == 384


@pytest.mark.parametrize("device", DEVICES)
def test_stable_diffusion_one_step(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch_dtype,
        safety_checker=None,  # speed‑up; not needed for a smoke test
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(0)
    prompt = "A tiny robot painting a watercolor landscape"

    image = pipe(
        prompt,
        generator=generator,
        num_inference_steps=1,
    ).images[0]

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"


@pytest.mark.parametrize("device", DEVICES)
def test_torch_matmul(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")

    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)

    c = torch.matmul(a, b)
    result = c[0, 0].item()

    assert result is not None


@pytest.mark.parametrize("device", DEVICES)
def test_whisper(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")

    model = whisper.load_model(
        "tiny", device=device, download_root=os.environ.get("WHISPER_HOME", None)
    )
    result = model.transcribe("test.wav")

    assert result is not None
