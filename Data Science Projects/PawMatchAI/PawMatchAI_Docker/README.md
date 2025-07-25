## 🚀 Getting Started
Welcome to PawMatchAI! This guide will walk you through setting up and running the project locally using Docker.
---
### 📋 Prerequisites
First, make sure you have the following tools installed on your system:
* [Git](https://git-scm.com/downloads)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
---
### ⚙️ Installation & Setup
Follow these steps to get the application up and running.

**1. Clone the Repository 📂**
Open your terminal and clone the entire `Learning-Record` repository. This command is the same for all operating systems.
```bash
git clone https://github.com/Eric-Chung-0511/Learning-Record.git
```

**2. Navigate into the Project Directory ➡️**
After cloning, you need to move into the correct project folder.
* **On macOS or Linux:**
    ```bash
    cd "Learning-Record/Data Science Projects/PawMatchAI"
    ```
* **On Windows (Command Prompt or PowerShell):**
    ```bash
    cd "Learning-Record\Data Science Projects\PawMatchAI"
    ```
***Note: All subsequent commands must be run from inside this `PawMatchAI` directory.***

**3. Download the Model File 🧠**
This project requires a pre-trained model file (`.pth`) that is hosted on Hugging Face Spaces.
* **Download Link:** [**Download `ConvNextV2Base_best_model.pth` from Hugging Face**](https://huggingface.co/spaces/DawnC/PawMatchAI/tree/main)
* **Placement:** After downloading, place the `.pth` file inside the `models/` directory. The final path should look like this:
    ```
    .../PawMatchAI/models/ConvNextV2Base_best_model.pth
    ```
---
### ⚠️ A Note on Dependencies and Security
Please be aware of the following regarding the project's dependencies:
* **Version Pinning:** The versions in `requirements.txt` have been carefully selected for maximum compatibility and stability, especially for deployment on platforms like Hugging Face Spaces which have specific version constraints. You may see security alerts from tools like GitHub's Dependabot regarding these pinned versions.
* **Security Risk (`torch.load`)**: The primary security alert relates to the `torch.load` function. The risk scenario involves a user loading a malicious, untrusted `.pth` file, which could potentially lead to remote code execution. This risk is mitigated in this project as it is designed to only load the trusted, pre-trained model file provided above. **For your safety, never load model files from untrusted sources.**
* **For Advanced Users**: Advanced users who understand the risks and wish to run in a more secure local environment are encouraged to fork this repository and upgrade the packages (e.g., `torch` to `2.6.0+` and other related libraries) in their own setup.
---
### 🖥️ GPU Support Configuration (Windows & Linux Users)

**Important Note: macOS users should skip this section and use the default CPU configuration.**

For Windows and Linux users with NVIDIA GPUs, you can enable GPU acceleration to significantly improve model inference speed.

**Prerequisites:**
- Install NVIDIA GPU drivers
- Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Verify Docker has proper GPU support configured

**Steps to Enable GPU Support:**

1. **Modify the Docker Compose Configuration**
   Replace the existing `services` section in your `docker-compose.yml` file with the following GPU-enabled configuration:

```yaml
services:
  pawmatch-ai:
    build: .
    container_name: pawmatch-ai-app
    ports:
      - "7860:7860"
    environment:
      - MODEL_PATH=/app/models/ConvNextV2Base_best_model.pth
      - ASSETS_PATH=/app/assets/example_images/
      - PYTHONPATH=/app
      - CUDA_VISIBLE_DEVICES=0  # Use the first GPU
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data:ro
      - ./assets:/app/assets:ro
    restart: unless-stopped
    networks:
      - pawmatch-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

2. **Verify GPU Availability**
   Test that the NVIDIA Container Toolkit is properly installed:
```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

**Performance Benefits:**
Enabling GPU support typically provides 3-10x faster model inference speeds, especially when processing high-resolution images or batch inference operations.

---
### ▶️ Build and Run the Application

**1. Build the Docker Image 📦**
Now, let's build the Docker image. This command reads the `Dockerfile` and creates a self-contained environment for the app. It might take a few minutes as it downloads all the necessary dependencies.
```bash
docker-compose build
```

**2. Run the Application ▶️**
Once the build is complete, start the application with this command. You will see logs in your terminal, indicating that the Gradio server has started.
```bash
docker-compose up
```
---
### 🎉 Access the Application
Once the container is up and running (you'll see a message like `Running on local URL: http://0.0.0.0:7860`), open your favorite web browser and navigate to:

👉 **http://localhost:7860**

That's it! You should now see the PawMatchAI interface and can start using the application. Enjoy! 🐾
