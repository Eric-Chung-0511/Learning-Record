## 🚀 Getting Started

Welcome to PawMatchAI! This guide will walk you through setting up and running the project locally using Docker.

### 📋 Prerequisites

First, make sure you have the following tools installed on your system:
* [Git](https://git-scm.com/downloads)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### ⚙️ Installation & Setup

Follow these steps to get the application up and running.

**1. Clone the Repository 📂**

Open your terminal and clone the entire `Learning-Record` repository.
```bash
git clone [https://github.com/Eric-Chung-0511/Learning-Record.git](https://github.com/Eric-Chung-0511/Learning-Record.git)
```

Then, navigate into the correct project directory:
```bash
cd "Learning-Record/Data Science Projects/PawMatchAI"
```
*Note: All subsequent commands must be run from this directory.*

**2. Download the Model File 🧠**

This project requires a pre-trained model file (`.pth`) that is hosted on Hugging Face Spaces.

* **Download Link:** [**Download `ConvNextV2Base_best_model.pth` from Hugging Face**](https://huggingface.co/spaces/DawnC/PawMatchAI/tree/main) ⬅️ 
* **Placement:** After downloading, place the `.pth` file inside the `models/` directory. The final path should look like this:
    ```
    .../PawMatchAI/models/ConvNextV2Base_best_model.pth
    ```

**3. Build the Docker Image 📦**

Now, let's build the Docker image. This command reads the `Dockerfile` and creates a self-contained environment for the app. It might take a few minutes as it downloads all the necessary dependencies.

```bash
docker-compose build
```

**4. Run the Application ▶️**

Once the build is complete, start the application with this command:
```bash
docker-compose up
```
You will see logs in your terminal, indicating that the Gradio server has started.

### 🎉 Access the Application

Once the container is up and running (you'll see a message like `Running on local URL: http://0.0.0.0:7860`), open your favorite web browser and navigate to:

👉 **http://localhost:7860**

That's it! You should now see the PawMatchAI interface and can start using the application. Enjoy! 🐾
