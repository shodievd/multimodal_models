# Multimodal AI Web Interface

This project implements a web interface for a multimodal AI system, featuring Visual Question Answering (VQA) using SmolVLM2 and Text-to-Image generation using Stable Diffusion.

## Prerequisites

- Docker
- Docker Compose (v2)
- NVIDIA GPU (optional, but recommended for performance) with NVIDIA Container Toolkit installed.

## Installation & Usage

### 1. Build the Docker Image

You can build the image locally:

```bash
docker-compose build
```

### 2. Run the Application

Start the application with Docker Compose:

```bash
docker-compose up -d
```

The application will be available at `http://localhost:5000`.

### 3. Configuration

You can configure the application using environment variables in `docker-compose.yml` or by creating a `.env` file.

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `PORT` | Port to expose on host | `5000` | Any valid port |
| `DEVICE` | Initial compute device | `cpu` | `cuda`, `cpu` |
| `MODEL_SIZE` | Model variant (if applicable) | `small` | `small`, `large` |

**Example: Running on GPU**

Ensure you have the NVIDIA Container Toolkit installed. The `docker-compose.yml` is already configured to use the GPU if available.

To force CPU mode if you have a GPU but don't want to use it, change `DEVICE` to `cpu` in `.env`.

### 4. Runtime Configuration

You can configure the model and device directly from the web interface:

1.  **Model Selection**: In the VQA chat interface, use the dropdown to select the model variant (e.g., "SmolVLM-Instruct").
2.  **Device Switching**: Use the dropdown in the chat header or the **Settings** menu to switch between CPU and GPU.
    -   *Note*: Switching devices moves the models in memory, which may take a few seconds.

### 5. Model Weights & Caching

The application uses a Docker volume to persist model weights on the host machine. This ensures models are downloaded only once.

- The cache is stored in the `./hf_cache` directory in the project root.
- To mount a pre-existing cache, modify the volume path in `docker-compose.yml`:

```yaml
volumes:
  - /path/to/your/cache:/root/.cache/huggingface
```

### 5. Changing the Port

To run on a different port (e.g., 8080), run:

```bash
PORT=8080 docker-compose up
```

Or modify the `ports` section in `docker-compose.yml`:

```yaml
ports:
  - "8080:5000"
```

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Models**:
    - VQA: `HuggingFaceTB/SmolVLM-Instruct`
    - Text-to-Image: `stabilityai/sdxl-turbo`

## Troubleshooting

- **OOM Errors**: If you run out of memory, try switching to CPU mode (slow) or ensuring no other processes are using the GPU.
- **Permission Issues**: Ensure the `./hf_cache` directory is writable by the docker user.
