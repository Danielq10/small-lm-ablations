# End-to-End Runbook: Remote Colab Training with Local MLflow Tracking

This runbook provides the exact step-by-step instructions to train models on a remote Google Colab GPU while streaming metrics, parameters, and model artifacts directly to your local workstation's MLflow server.

MLflow traffic travels through an **SSH reverse tunnel**: Colab talks to `http://127.0.0.1:5000` on its side, and the tunnel delivers it to `127.0.0.1:5000` on your machine. Nothing is exposed publicly. Training data is read from **Google Drive**, not uploaded from your machine.

---

## Prerequisites

- Local machine running Linux, or **WSL (Ubuntu)** on Windows, with your `small-lm-ablations` workspace.
  - The `colab` CLI does not run on native Windows (`No module named 'termios'`), and Windows OpenSSH has no `ControlMaster` support. Run every "Local Machine" command below inside WSL, including `mlflow server`, so the tunnel's `127.0.0.1:5000` reaches it. Your Windows browser can still open `http://localhost:5000`, because WSL2 forwards localhost.
- `colab` CLI installed and authenticated.
- An Ed25519 key at `~/.ssh/id_ed25519` (Colab rejects RSA keys).
- GitHub repository: `https://github.com/Danielq10/small-lm-ablations.git`
- `train.bin` and `val.bin` uploaded to a folder in your Google Drive.
- Google Colab account with GPU runtime access.

---

## Step 0: SSH Config (Local Machine, One-Time)

Add this block to `~/.ssh/config`:

```
Host colab
    HostName colab-runtime
    User root
    IdentityFile ~/.ssh/id_ed25519
    ProxyCommand colab ssh --proxy-mode -s trainer
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    ServerAliveInterval 30
    ServerAliveCountMax 4
    ControlMaster auto
    ControlPath ~/.ssh/cm-%C
    ControlPersist yes
```

> [!NOTE]
> A Colab runtime accepts **one SSH connection at a time** (a second one is rejected with HTTP 429). `ControlMaster` makes the tunnel, your shell and any `scp` share a single connection. While the tunnel is up, connect with `ssh colab`, **not** `colab ssh -s trainer`: the latter opens its own connection and will be refused.
>
> The host-key options match what `colab ssh` uses itself, because every new runtime has a new host key.

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

## Step 2: Start Colab Session (Local Machine)

Start your Google Colab instance with an **L4** (or **A100**) GPU:

```bash
colab new -s trainer --gpu L4
```

> [!WARNING]
> Do not use `--gpu T4` for the 27M ablation configs (`batch_size: 128`). Baseline peaks at ~14.8 GB VRAM and SwiGLU/RoPE variants allocate more activation memory (likely OOM on T4's 15 GB), and T4 lacks TF32 tensor cores.

---

## Step 3: Open the SSH Tunnel (Local Machine, Terminal A)

In a dedicated local terminal (Terminal A), open the shared SSH connection with the reverse port forward, and reconnect automatically if it drops:

```bash
until colab status -s trainer | grep -q "not found"; do
  ssh -N -o ControlPersist=no -o ExitOnForwardFailure=yes \
      -R 5000:127.0.0.1:5000 colab
  echo "[tunnel] dropped - reconnecting in 5s"; sleep 5
done
```

- `-R 5000:127.0.0.1:5000` makes Colab's `127.0.0.1:5000` lead to your local MLflow server.
- `-N` opens no shell. This connection only carries the tunnel and acts as the shared master for Step 5.
- `ExitOnForwardFailure=yes` makes a reconnect fail loudly (and retry) while Colab still holds port 5000 from the dropped connection, instead of connecting without a tunnel.
- The `colab status` guard stops the loop only once the CLI reports the `trainer` session as not found. Otherwise the `ProxyCommand` would auto-create a new (CPU) runtime called `trainer`. A status check that fails, for example during a network blip, keeps the loop retrying.

Leave Terminal A running for the whole training run.

---

## Step 4: Mount Google Drive (Local Machine, Terminal B)

In a second local terminal (Terminal B):

```bash
colab drivemount -s trainer
```

If prompted, open the printed authorization URL, grant access, then press `Enter`. Drive is mounted at `/content/drive`, so your files are under `/content/drive/MyDrive/`.

---

## Step 5: SSH into Colab & Start `tmux` (Terminal B)

```bash
ssh colab
```

This reuses the tunnel connection from Step 3. You land in `/root`, so move to Colab's working directory and **start a `tmux` session** so training survives accidental disconnections:

```bash
cd /content
tmux new -s train
```

---

## Step 6: Clone Repository & Copy Data from Drive (On Colab)

Make sure your latest local commits are pushed to `origin/main` first, then clone the repository and navigate into it:

```bash
git clone https://github.com/Danielq10/small-lm-ablations.git && cd small-lm-ablations
```

Copy the datasets from Drive onto the runtime's local disk:

```bash
# Set to the Drive folder that holds train.bin and val.bin
DRIVE_DATA=/content/drive/MyDrive/<your-data-folder>

ls -lh "$DRIVE_DATA"/train.bin "$DRIVE_DATA"/val.bin
cp -v "$DRIVE_DATA"/train.bin "$DRIVE_DATA"/val.bin data/
ls -lh data/
```

> [!NOTE]
> Copy the files instead of symlinking them. `train.py` reads the `.bin` files through `np.memmap`, and reading over the Drive FUSE mount is slower and can break if the mount drops during a long run.
>
> `data/polish_bpe_8k.json` comes from the git clone. Make sure the committed version is the tokenizer that produced the `.bin` files on Drive.
>
> To send any other local file, `scp` uses the shared connection too: `scp <file> colab:/content/small-lm-ablations/<dest>/`

---

## Step 7: Set MLflow Environment Variable (On Colab)

Inside your Colab `tmux` session, point MLflow at the tunnel:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

### Verify Connectivity

Ping your local MLflow server from Colab:

```bash
curl -s "${MLFLOW_TRACKING_URI}/health"
```

*Expected output: `OK`*

If this fails, check that Terminal A shows no errors and that the MLflow server from Step 1 is running.

> [!IMPORTANT]
> **Artifact Proxying Requirement**: Make sure the target experiment (e.g. `arch-ablation-final` or `smoke-tests`) is created through your running `mlflow server` so its `artifact_location` is `mlflow-artifacts:/...`. `train.py` enforces this at startup and fails fast if an HTTP run targets a legacy local-path experiment (or if a local `sqlite:///` run targets a proxied experiment — when running locally against `arch-ablation-final` or `smoke-tests`, set `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`).

---

## Step 8: Install the Required Python Packages (On Colab)

```bash
uv sync
```

---

## Step 9: Run Model Training (On Colab)

First run a quick smoke check to confirm metrics, LLM traces, and `.pt` artifacts arrive locally in `runs/artifacts/`:

```bash
uv run python train.py --config base_tiny --test
```

Then launch the full ablation runs:

```bash
uv run python train.py --config base
```

*(Available ablation configs: `base`, `rope`, `gqa`, `swiglu`, `rope_gqa`, `modern`, `rope_swiglu`; smoke configs: `base_tiny`, `test_config`.)*

### Monitoring & Session Controls

- **Monitor GPU in split pane**:
  - Press `Ctrl + b`, then `"` (splits screen horizontally).
  - Run `watch -n 1 nvidia-smi` in the new pane.
  - Press `Ctrl + b`, then `Up Arrow` to return to training logs.
- **Detach and let it run**:
  - Press `Ctrl + b`, then `d`.
  - You can close Terminal B. **Keep Terminal A (the tunnel), the MLflow server and your machine awake.** Every metric and checkpoint goes through the tunnel to your local server. MLflow's HTTP client retries with backoff, so a short reconnect in Terminal A is absorbed. An outage longer than that raises in `train.py` and ends the run.
- **Reattach later** (Terminal B):
  ```bash
  ssh colab
  tmux attach -t train
  ```
- **Live Local Dashboard**:
  - Check `http://localhost:5000` on your workstation. Training loss, validation loss, and hyperparameters will update in real time.
- **Model Checkpoints**:
  - When training completes, the model artifact and `.pt` checkpoint are automatically synced into `runs/artifacts/` on your local workstation.

---

## Teardown (When Finished)

1. Inside Colab, run `exit` twice: once to leave tmux, once to leave SSH.
2. In Terminal A, stop the tunnel loop with `Ctrl + C`. Do this **before** step 3, so the loop cannot reconnect and recreate the runtime.
3. Free the GPU runtime to save compute units (local machine):
   ```bash
   colab stop -s trainer
   ```
4. Clear any leftover shared SSH connection (local machine):
   ```bash
   ssh -O exit colab 2>/dev/null
   ```
