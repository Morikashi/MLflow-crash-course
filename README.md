# Practical MLflow: Step-by-Step Notebooks

![MLFlow logo](https://www.the-odd-dataguy.com/images/posts/20191113/cover.jpg)

Welcome to a hands-on journey into the world of modern Machine Learning Operations (MLOps) and Generative AI, all orchestrated with the power of **MLflow**! This series of 8 (+1 bonus) Jupyter notebooks is designed to take you from the fundamentals of MLflow to building, evaluating, and managing complex GenAI applications, step by step.

Whether you're new to MLflow or looking to deepen your understanding of its application in cutting-edge ML and GenAI scenarios, these notebooks offer practical, code-first examples. We'll tackle real-world(-inspired) problems, use recent datasets from Hugging Face, and integrate with a modern stack including **LangChain, LangGraph, Ollama, PEFT, TRL, Pydantic, and more!**

---

## ✨ What You'll Explore ✨

This series is structured to build your skills progressively:

1.  **Foundations of MLflow:** Mastering experiment tracking and hyperparameter optimization.
2.  **Productionizing Models:** Moving beyond experiments to model registry and local deployment.
3.  **Generative AI with MLflow:** Applying MLOps principles to RAG, LLM fine-tuning, evaluation, and agentic systems.

![MLFlow Workflow](https://mlflow.org/docs/latest/assets/images/learn-core-components-b2c38671f104ca6466f105a92ed5aa68.png)

---

## 📚 The Notebook Journey 📚

Here's a glimpse into each adventure:

### 📒 Notebook 1: MLflow 101 - Experiment Tracking with Modern ML Pipelines
*   **Overview:** Get started with MLflow's core capability: experiment tracking. We'll train an XGBoost model on the California Housing dataset and meticulously log parameters, metrics, and model artifacts.
*   **MLflow Focus:** `mlflow.start_run()`, `mlflow.log_param()`, `mlflow.log_metric()`, `mlflow.log_artifact()`, `mlflow.set_experiment()`, MLflow UI basics.
*   **Tech & Concepts:** XGBoost, Scikit-learn, Pandas, Hugging Face Datasets, Regression.
*   **Outcome:** Confidently track and compare your ML experiments using the MLflow UI.
    ![MLFlow Tracking](https://media.datacamp.com/cms/google/ad_4nxekg7ftko2m1hrkr-bwr-kq5gzr9wfugs9spjvgmoca-yykxhhepgcwxxo9yrbhu4barnqvmx6psn9scgku1car3lvlhltqnada0i9m7cg_glbdf5ty3lu4t3pcyxel6dyh1n84fcsl3xqvgdktujpvrian.png)

### 🛠️ Notebook 2: Advanced Hyperparameter Optimization & Model Selection
*   **Overview:** Dive into systematic hyperparameter optimization using Optuna integrated with MLflow. We'll optimize our XGBoost model for the California Housing dataset and track parent/child runs.
*   **MLflow Focus:** Nested runs (`mlflow.start_run(nested=True)`), logging HPO study parameters, comparing trials in MLflow UI, retrieving best run parameters.
*   **Tech & Concepts:** Optuna, XGBoost, Hyperparameter Optimization (TPE), Model Selection.
*   **Outcome:** Efficiently find optimal hyperparameters and manage complex HPO studies.

### 🚢 Notebook 3: MLflow Model Registry & Production Deployment
*   **Overview:** Learn how to manage the lifecycle of your models using the MLflow Model Registry. We'll register our optimized XGBoost model, manage its versions and aliases, and deploy it locally as a REST API.
*   **MLflow Focus:** `mlflow.register_model()`, Model Versioning, Aliases (`@champion`), Tags, `mlflow.pyfunc.load_model()`, `mlflow models serve`, `MlflowClient` API.
*   **Tech & Concepts:** Model Lifecycle Management, Local REST API Deployment (FastAPI/MLServer basics).
*   **Outcome:** Understand how to version, stage, and deploy production-ready models.
    *(Imagine a gif here showing model stage transition or a local API call)*

### 🧠 Notebook 4: Fine-Tuning Qwen3-0.6B with MLflow - Custom Domain Adaptation
*   **Overview:** Step into the world of LLM customization! We fine-tune the efficient `Qwen/Qwen3-0.6B` model on a recipe generation dataset using Parameter-Efficient Fine-Tuning (PEFT) with LoRA and Hugging Face TRL.
*   **MLflow Focus:** Tracking fine-tuning parameters (LoRA config, training args), metrics (training loss), and saving/logging LoRA adapters as artifacts.
*   **Tech & Concepts:** LLMs (Qwen3-0.6B), Supervised Fine-Tuning (SFT), PEFT, LoRA, Hugging Face `transformers`, `peft`, `trl`, BitsAndBytes (Quantization), Domain Adaptation.
*   **Dataset:** `alignment-handbook/recipe_instructions`.
*   **Outcome:** Adapt a powerful, small LLM to a specific creative task and manage the fine-tuning lifecycle.

### 📊 Notebook 5: Evaluating and Benchmarking LLMs with MLflow
*   **Overview:** This notebook was **conceptually replaced by Notebook 4 (Qwen3 fine-tuning)**. The original intent was to fine-tune Phi-3. The Qwen3 notebook serves as a more current example of LLM fine-tuning. The evaluation aspects originally planned for a later notebook are brought forward.

*(Self-correction from our development: The original sequence had Phi-3 fine-tuning here. We've made Notebook 4 the Qwen3 fine-tuning, which covers this topic with a newer model. The next logical step after fine-tuning is robust evaluation.)*

### 🥇 Notebook 6: Evaluating and Benchmarking LLMs with MLflow (Focus on Summarization)
*   **Overview:** How good are our LLMs (base vs. fine-tuned)? We systematically evaluate different models (e.g., base Qwen3-0.6B, our recipe-fine-tuned Qwen3, and a baseline like Flan-T5-small) on a text summarization task using Hugging Face `evaluate` and `mlflow.evaluate()`.
*   **MLflow Focus:** `mlflow.evaluate()`, logging standard NLP metrics (ROUGE, BERTScore), comparing evaluation runs in the MLflow UI to create a benchmark.
*   **Tech & Concepts:** LLM Evaluation, Text Summarization, ROUGE, BERTScore, Hugging Face `evaluate` library.
*   **Dataset:** `openai/summarize_from_feedback` (TLDR subset).
*   **Outcome:** Objectively compare LLM performance and make data-driven model selection decisions.
    ![MLFlow UI](https://blog.min.io/content/images/2025/03/Screenshot-2025-03-10-at-3.30.33-PM.png)

### 🛠️ Notebook 7: Tool-Calling Agents with LangGraph, Ollama, and MLflow
*   **Overview:** Build an AI agent that can intelligently decide to use external tools (e.g., mock weather service, calculator) to answer user queries. We'll use LangGraph for agent orchestration and Ollama with `phi3:mini` or `Qwen3-0.6B` for local LLM power.
*   **MLflow Focus:** `mlflow.langchain.autolog()` for tracing LangGraph executions, visualizing agent decision paths, LLM calls, and tool invocations in the MLflow UI.
*   **Tech & Concepts:** AI Agents, Tool Use, LangGraph (`StateGraph`, Nodes, Conditional Edges), Ollama, Local LLMs (`phi3:mini` or `Qwen3-0.6B`).
*   **Outcome:** Create and trace dynamic, tool-using agents.
    *(Imagine a gif showing a decision graph or an agent using a tool)*

### 🤝 Notebook 8: Advanced Function-Calling and Agent2Agent Protocols in LLM Apps
*   **Overview:** Enhance our agent's tool-calling capabilities using Pydantic for robust function argument schemas. We'll then design a hierarchical multi-agent system (e.g., Supervisor coordinating Researcher and Writer agents) using LangGraph for a content generation task.
*   **MLflow Focus:** Tracing complex multi-step, multi-component agent interactions, visualizing data flow and decisions between different "agentic" parts.
*   **Tech & Concepts:** Advanced Function Calling, Pydantic, Multi-Agent Systems (Hierarchical), LangGraph, Ollama. Conceptual overview of A2A protocols (MCP).
*   **Outcome:** Design, build, and trace more sophisticated collaborative agent systems.

### 📈 Notebook 9: Custom Metrics and Evaluation for Generative Tasks
*   **Overview:** Go beyond standard metrics for generative AI. Define and implement custom heuristic-based metrics (e.g., length constraints, keyword presence) and explore LLM-as-a-Judge metrics for qualities like "helpfulness" or "coherence," integrating them with `mlflow.evaluate()`.
*   **MLflow Focus:** `mlflow.metrics.make_metric`, `mlflow.evaluate()` with `extra_metrics`, logging custom scores and detailed per-sample evaluation tables.
*   **Tech & Concepts:** Custom Evaluation, Heuristic Metrics, LLM-as-a-Judge, Generative AI Quality Assessment.
*   **Dataset:** `openai/summarize_from_feedback` (revisited for custom evaluation context).
*   **Outcome:** Develop and apply tailored evaluation strategies for a deeper understanding of your generative models.

---

## 🛠️ Prerequisites & Setup 🛠️

1.  **Python Environment:** It's highly recommended to use a virtual environment (e.g., `conda` or `python -m venv`). Python 3.9+ is advisable.
2.  **Dependencies:** Each notebook will list its specific Python package requirements at the top. You can typically install them using `pip install -r requirements.txt` (if a global one is provided) or `pip install <package_name>` as you go.
3.  **Ollama (for local LLMs):**
    *   Download and install Ollama from [ollama.com](https://ollama.com/).
    *   Ensure the Ollama application/server is running before executing notebooks that use it.
    *   Pull the required models (e.g., `ollama pull phi3:mini`, `ollama pull qwen3:0.6b`) via your terminal.
4.  **MLflow:** Ensure MLflow is installed (`pip install mlflow`). The MLflow UI can be launched by running `mlflow ui` in your terminal from the root directory of this repository (where the `mlruns` folder will be created).
5.  **Hugging Face Account (Optional):** For downloading certain models/datasets or pushing your fine-tuned adapters to the Hub, you might need a Hugging Face account and to be logged in via `huggingface-cli login`.

---

## 🚀 How to Use This Repository 🚀

1.  **Clone the Repository:**
    ```
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Navigate to Notebooks:** Each notebook is self-contained or builds upon concepts from previous ones.
3.  **Install Dependencies:** Check the first few cells of each notebook for `pip install` commands for required libraries.
4.  **Run Jupyter:** Launch Jupyter Lab or Jupyter Notebook and open the desired `.ipynb` file.
    ```
    jupyter lab
    # or
    jupyter notebook
    ```
5.  **Experiment!** Feel free to modify the code, try different models, datasets, or parameters. The goal is learning through doing!

---

## 💡 Key Learnings Across the Series 💡

By completing this series, you will gain:
*   **Comprehensive MLflow Proficiency:** From basic tracking to advanced model registry, LLM evaluation, and tracing complex agentic systems.
*   **Modern MLOps Practices:** Understand how to apply MLOps principles to both traditional ML and cutting-edge Generative AI workflows.
*   **Hands-On Experience with GenAI Stack:** Practical skills in using Hugging Face Transformers, PEFT, TRL, LlamaIndex, LangGraph, Pydantic, and Ollama.
*   **Problem-Solving Skills:** Approach complex ML/GenAI tasks by breaking them down and leveraging the right tools.

---

## 🌟 Further Exploration & Contribution 🌟

*   Explore the official [MLflow Documentation](https://mlflow.org/docs/latest/index.html) for even more features.
*   Try different models, datasets, or tasks in each notebook.
*   Extend the agents with more tools or more sophisticated logic.
*   Contribute your ideas or improvements! (Details on how to contribute can be added here).
