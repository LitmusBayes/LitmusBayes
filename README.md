# LitmusBayes: Accelerating Memory Consistency Testing via Bayesian Optimization 

## 1. LitmusBayes Framework

We aim to contribute a novel parameter-aware framework for accelerating memory consistency testing of litmus tests to the hardware verification community. We release the complete source code of LitmusBayes that covers the whole automated testing pipeline, including Hybrid Two-Tower Embedding, Contextual Bayesian Optimization, and Cross-Platform Transfer Learning. We hope the publicly available implementation could facilitate follow-up research towards scaling litmus-test verification across diverse chips without relying on exhaustive per-test tuning.

## 2. Reproduction

### 2.1. Environment

The implementation and evaluation of LitmusBayes are performed with:
* **OS:** Linux (e.g., Ubuntu 22.04)
* **Language:** Python 3
* **Hardware:** At least 16GB RAM, XuanTie C910 development board (optional), SpacemiT K1 development board (optional)

### 2.2. Configuration

To configure the physical development boards for hardware-based reproduction, set the corresponding environment variables and ensure the target directories exist on the boards:

```bash
export K1_HOST="10.42.0.58"
export K1_PORT="22"
export K1_USERNAME="root"
export K1_PASSWORD="bianbu"
export K1_REMOTE_PATH="/root/test"

export C910_HOST="10.42.0.58"
export C910_PORT="22"
export C910_USERNAME="sipeed"
export C910_PASSWORD="sipeed"
export C910_REMOTE_PATH="/root/test"
```
*(Note: This step can be skipped if you are reproducing experiments using the pre-collected execution logs.)*

### 2.3. Installation

To install the required dependencies for the LitmusBayes framework:

```bash
pip install -r requirements.txt
```

### 2.4. Execution

To run the reproducibility experiments based on pre-collected execution logs (without physical hardware):

```bash
cd src/litmusbayes/bayes/experiment
python3 run.py 
```

To run the reproducibility experiments directly on the configured C910 and SpacemiT K1 development boards (building on previously completed model pre-training):

```bash
cd src/litmusbayes/bayes/experiment
python3 run_from_hardware.py
python3 run.py 
```