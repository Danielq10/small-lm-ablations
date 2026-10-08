# End-to-End Runbook: Remote Colab Training with Local MLflow Tracking

This runbook provides the exact step-by-step instructions to train models on a remote Google Colab GPU while streaming metrics, parameters, and model artifacts directly to your local workstation's MLflow server.

---

## Prerequisites

- Local machine running Linux with your `small-lm-ablations` workspace.
- GitHub repository: `https://github.com/Danielq10/small-lm-ablations.git`
- Google Colab account with GPU runtime access.

---

## Step 1: Launch MLflow Tracking Server (Local Machine)

On your local workstation inside the `small-lm-ablations` project folder, start the MLflow server:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///runs/mlflow.db \
  --artifacts-destination ./runs/artifacts \
  --host 127.0.0.1 \
  --port 5000
```

> [!NOTE]
> The `--artifacts-destination ./runs/artifacts` flag instructs the server to proxy all artifact uploads. Colab will upload `.pt` checkpoints and models over HTTP directly into your local `runs/artifacts/` directory.

Keep this terminal tab open, or run it inside a local `tmux` session. Verify it is running by visiting `http://localhost:5000` in your local web browser.

---

## Step 2: Open / Forward Port in VS Code (Local Machine)

1. In VS Code, navigate to the bottom dock and click on the **Ports** tab (next to Terminal/Output), or press `Ctrl + Shift + P` and search for:

   ```bash
   Ports: Focus on Ports View
   ```

2. Click **Forward a Port** (or the `+` button).
3. Type `5000` and press `Enter`.
4. Right-click the newly forwarded port `5000` in the list:
   - Select **Port Visibility** -> choose **Public**.
5. Copy the URL generated under the **Forwarded Address** column (e.g. `https://xxxx-5000.use.devtunnels.ms` or `http://localhost:5000` if using SSH remote port forwarding).

---

## Step 3: Start Colab Session

Start your Google Colab instance (with GPU enabled):

  ```bash
  colab start -s trainer --gpu T4
  ```

---

## Step 4: SSH into Colab Session

Connect to your running Colab instance from your terminal:

```bash
colab ssh -s trainer
```

Once inside Colab, **start a `tmux` session** so training survives accidental disconnections:

```bash
tmux new -s train
```

---

## Step 5: Set MLflow Environment Variable (On Colab)

Inside your Colab terminal (`tmux` session), point MLflow to your forwarded port address:

```bash
# Replace with the Forwarded Address from Step 2
export MLFLOW_TRACKING_URI="<YOUR_FORWARDED_PORT_URL>"
```

### Verify Connectivity

Ping your local MLflow server from Colab:

```bash
curl -s "${MLFLOW_TRACKING_URI}/health"
```

*Expected output: `OK`*

---

## Step 6: Clone Repository & Install Dependencies (On Colab)

Clone your repository and navigate to the project directory:

```bash
git clone https://github.com/Danielq10/small-lm-ablations.git
cd small-lm-ablations
```

## Step 7: Loogout from the session and upload training data

```bash
scp -o 'ProxyCommand=colab ssh --proxy-mode -s trainer' -i ~/.ssh/id_ed25519 data/train.bin data/val.bin data/polish_bpe_8k.json root@colab:/content/small-lm-ablations/data/
```

## Step 8: SSH to the session again

## Step 9:Install the required Python packages on Colab

```bash
uv sync
```

---


## Step 10: Run Model Training (On Colab)

Launch the training loop with your desired configuration:

```bash
python train.py --config chinchilla_t4
```

*(Or use `--config base` / `--config base_tiny`).*

### Monitoring & Session Controls

- **Monitor GPU in split pane**: 
  - Press `Ctrl + b`, then `"` (splits screen horizontally).
  - Run `watch -n 1 nvidia-smi` in the new pane.
  - Press `Ctrl + b`, then `Up Arrow` to return to training logs.
- **Detach and let it run**:
  - Press `Ctrl + b`, then `d`.
  - You can now safely close your SSH session or shut your laptop.
- **Reattach later**:
  - SSH back into Colab and run:
    ```bash
    tmux attach -t train
    ```
- **Live Local Dashboard**:
  - Check `http://localhost:5000` on your workstation. Training loss, validation loss, and hyperparameters will update in real time.
- **Model Checkpoints**:
  - When training completes, the model artifact and `.pt` checkpoint are automatically synced into `runs/artifacts/` on your local workstation.

---

## Teardown (When Finished)

1. Inside Colab, exit tmux:
   ```bash
   exit
   ```
2. Free the GPU runtime to save compute units:
   ```python
   from google.colab import runtime
   runtime.unassign()
   ```
