<div align="center">
<h1>Ai wise council</h1>
<h3>A sophisticated AI system that leverages multiple LLMs to provide the best possible responses through a council-based approach👩‍⚖️👨‍⚖️</h3>
</div>

<div align="center">
<a href="https://contrastoai.com/" target="_blank">
<img alt="Contrasto.AI Logo" src="./data/images/logo_contrasto.png" height="50">
</a>
<p><a href="https://contrastoai.com/" target="_blank">Contrasto.AI</a></p>
</div>

<br/>

<div align="center">

   <a href="https://www.linkedin.com/company/contrastoai" target="_blank"> <img alt="Follow us on LinkedIn" src="https://img.shields.io/badge/LinkedIn-Follow%20us-blue?logo=linkedin">
   ![GitHub branch check runs](https://img.shields.io/github/check-runs/AnaBelenBarbero/ai-wise-council/deploy?label=Deployed%20to%20GCP)

</a></div>
<br/>

## Table of Contents
- [Overview](#overview)
- [Components](#components)
- [Installation](#installation)

## Overview

The challenge of AI oversight

As AI systems become increasingly sophisticated, **we face a critical challenge in ensuring their proper behavior**. Currently, human review and data labeling are the primary methods for training AI models - a requirement even mandated by the EU AI Act. However, this raises an important question: As AI systems surpass human knowledge, how can we effectively validate their performance? Will human oversight remain sufficient? or should we rely on AI models to oversee more advanced ones?

Our approach: AI wise council

AI wise council addresses this challenge through an innovative approach to AI decision-making. Drawing inspiration from **Anthropic's research on [scalable oversight](https://arxiv.org/pdf/2402.06782)**, we've developed a virtual council system that combines multiple specialized AI models. 

Here's how it works:

- 2 AI models (randomly selected from OpenAI, Anthropic, DeepSeek) serve as expert debaters.
- The debaters receive a context and a query about it. 
- Each model contributes its unique expertise and perspective regarding the query. One model is a truth-telling council member while another is a deceptive council member.
- We have fintuned a weak Judge model (phi-3-mini) to oversee the process. This model does not receive the context, only the debaters' arguments. 
- Finally, the Judge model evaluates the debaters' arguments and detects who is telling the truth.

To see the dynamic of the debate, check the `01_agentic_debate.ipynb` notebook.

With this approach, we achieve more reliable and thoughtful AI outputs while addressing the fundamental challenge of AI oversight!

*This projects is part of the 2st capstone of [Datatalks Machine Learning Engineering zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master).*

## Components

### Core scripts
- `train.py`: Model training pipeline with dataset preparation.
- `finetuning.py`: Fine-tuning pipeline and model upload to Hugging Face.
- `predict.py`: FastAPI server for real-time model predictions.

### Components directory
Core system components using Langchain and Langgraph:
- `agents/`: AI debate participants implementation.
  - Truth-telling council member.
  - Deceptive council member.
  - Judge that evaluates arguments and detects deception.
- `debate/`: Debate flow using Langgraph StateGraph.
  - Controls turn-taking via conditional edges.
  - Manages debate progression to conclusion.
- `model_client/`: Selection of the model provider.

### Prompts Directory  
System prompts and agent instructions:
- `constructors/`: Role-based prompt templates.
- `instructions/`: Specific agent behavior guidelines.

### Notebooks
- `01_agentic_debate.ipynb`: Debate system workflow demonstration.
- `02_generate_dataset.ipynb`: Fine-tuning dataset generation.
- `03_EDA.ipynb`: Exploratory Data Analysis.

### StateGraph Visualization
![StateGraph](data/images/stategraphlg.png)

The StateGraph above illustrates the debate flow between agents, showing:
- Initial random selection between good and bad faith agents.
- Turn-taking debate cycle with conditional transitions.
- Final judgment phase when debate concludes.


## Installation

### 1. Prerequisites 📑✅
- Python 3.12
- Poetry for dependency management and virtual environment
  - Install from [python-poetry.org](https://python-poetry.org/)
- Make for running commands and scripts
  - Provides shortcuts defined in `Makefile`
  - Install via your system's package manager
- API Keys (add to `.env` file)
  - Hugging Face token (`HF_TOKEN`) for model management.
  - OpenAI API key.
  - Anthropic API key.  
  - DeepSeek API key.
  - Copy `.env_example` to `.env` and add your keys.
- For training:
  - NVIDIA GPU with minimum 32GB VRAM (e.g. A6000).
  - CUDA 12.1 installed.
  - At least 32GB system RAM
  - 40GB free disk space

We have trained the model on [Runpod](https://www.runpod.io/) in A6000 GPU during few hours for finetuning. 

### 2. Training Steps 🏋️‍♀🏋

#### A. Pre-trained transfer learning

This step loads the pretrained `phi-3-mini` model as a starting point for fine-tuning.

Key features of Phi-3 mini:
- 3.8B parameters.
- Lightweight, state-of-the-art open model.
- Learn more: [Phi-3 Mini Documentation](https://ollama.com/library/phi3:mini).

This model represents the "weak judge" who is going to supervise the debate and detect the debater that is telling the truth.

#### B. Fine tune: data generation and hyperparameter tunning

We have generated a dataset of debates (with 3 rounds of interactions). In each debate, we randomly select 2 models to act as debaters (OpenAI, Anthropic and DeepSeek). The data generation process is described in `02_generate_dataset.ipynb`.

After some EDA and preprocessing (described in `03_EDA.ipynb`), we have performed hyperparameter tuning to optimize the model’s performance.

##### C. Run the training

The default entrypoint as we can see it in the Dockerfile is the finetuning process `poetry run python ai_wise_council/finetuning.py`.

The training process can be executed either locally or on any GPU provider like RunPod, depending on your hardware resources, specified above.

To train on RunPod:
1. Create a RunPod account and set up a CUDA GPU for training.
2. Build and push the Docker image to the Docker Hub
3. Create a template in RunPod with the Docker image, resources, ENVARS and volume persistence.
4. Run the instance and automatically the finetune process will be executed, the model will be uploaded to Hugging Face and stored in `CACHE_DIR`.

<div align="center">
<img alt="runpod" src="../data/images/runpod.jpeg" height="200">
</div>

#### D. Upload trained model to Hugging Face

Once training is complete, we upload the [saved model](https://huggingface.co/ana-contrasto-ai/ai-wise-council/tree/main) to Hugging Face for easy deployment and sharing. 

### 3. Deployment Steps 🐳🐳
Clone the repository:

   ```bash
   gh repo clone AnaBelenBarbero/ai-wise-council
   cd ai-wise-council
   ```

#### A. Run local venv
1. Install dependencies:

    ```bash
    poetry install
    ```

2. Run FastAPI:

    ```bash
    make run-api-dev
    ```

#### B. Run local Docker container
1. Run the train container dev:

    ```bash
    make run-docker-dev
    ```

2. Run the predictcontainer, mapped to `localhost:80`:

    ```bash
    make docker-build-predict
    ```

Check the first part of `04_call_api.ipynb` for local inference testing.

#### C. GCP Cloud run deployed
This time the deploy is not handled with an specific Github branch, but by pushing newversions of the image to the DOCKER HUB. 

<div align="center">
<img alt="dockerhub" src="../data/images/dockerhub.png" height="200">
</div>

Then with a GCP Cloud Run instance, the image will be extracted and deployed automatically. The entrypoint is overridden with the predict FastAPI server.

Check the second part of `04_call_api.ipynb` for GCP authentication and request-building details.
