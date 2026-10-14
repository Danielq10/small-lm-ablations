import time

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table


class TrainingDisplay:
    def __init__(
        self,
        cfg,
        vocab_size,
        embedding_params,
        model_params,
        training_tokens,
        validation_tokens,
        training_steps,
        expected_training_time,
        optimal_model_params,
        expected_loss,
    ):
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.embedding_params = embedding_params
        self.model_params = model_params
        self.training_tokens = training_tokens
        self.validation_tokens = validation_tokens
        self.training_steps = training_steps
        self.expected_training_time = expected_training_time
        self.optimal_model_params = optimal_model_params
        self.expected_loss = expected_loss
        self.console = Console()

        self.pbar = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn(
                "loss {task.fields[loss]} | val {task.fields[val]} | {task.fields[tps]} tok/s"
            ),
            console=self.console,
            expand=True,
        )

    def print_header(
        self,
    ):
        model_table = Table(
            title="MODEL SUMMARY",
            title_style="#f2542d",
            show_header=False,
            title_justify="center",
            box=box.ROUNDED,
            show_edge=True,
        )
        model_table.add_column(
            header_style="bold green",
            style="bold cyan",
            justify="left",
            no_wrap=True,
        )
        model_table.add_column(
            style="magenta",
            justify="center",
            no_wrap=True,
        )
        model_rows = [
            ("Vocab size", self.vocab_size),
            ("n_embed", self.cfg.model.n_embed),
            ("block_size", self.cfg.model.block_size),
            ("n_blocks", f"{self.cfg.model.n_blocks}"),
            ("n_heads", f"{self.cfg.model.n_heads}"),
            ("dropout", self.cfg.model.dropout),
            ("pos_encoding", self.cfg.model.pos_encoding),
            ("Embedding params", f"{self.embedding_params / 1e6:.2f} M"),
            ("Model params", f"{self.model_params / 1e6:.2f} M"),
            (
                "Total params",
                f"{(self.embedding_params + self.model_params) / 1e6:.2f} M",
            ),
            ("Tokens / param", f"{self.training_tokens / self.model_params:.2f}"),
        ]

        for k, v in model_rows:
            model_table.add_row(k, str(v))

        calculated_eval_interval = max(1, self.training_steps // 20)

        training_table = Table(
            title="TRAINING CONFIGURATION",
            title_style="#f2542d",
            show_header=False,
            title_justify="center",
            box=box.ROUNDED,
            show_edge=True,
        )
        training_table.add_column(
            style="bold cyan",
            justify="left",
            no_wrap=True,
        )
        training_table.add_column(
            style="#f5dfbb",
            justify="center",
            no_wrap=True,
        )

        training_rows = [
            (
                "Train tokens",
                f"{self.training_tokens / 1e6:.2f} M",
            ),
            (
                "Val tokens",
                f"{self.validation_tokens / 1e6:.2f} M",
            ),
            ("Batch size", self.cfg.train.batch_size),
            (
                "Training Steps",
                f"{self.training_steps}",
            ),
            (
                "Eval interval",
                f"{calculated_eval_interval}",
            ),
            (
                "Expected loss / hours",
                f"{self.expected_loss:.2f} / {self.expected_training_time:.2f}",
            ),
            ("Optimal model params", f"{self.optimal_model_params / 1e6:.2f} M"),
        ]
        for k, v in training_rows:
            training_table.add_row(k, str(v))

        grid = Table.grid(expand=False, padding=(0, 4))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(model_table, training_table)
        self.console.print()
        self.console.print(grid)

    def start_training(self):
        self.pbar.start()
        self.training_task = self.pbar.add_task(
            description="[green]Training",
            total=self.training_steps,
            loss="-",
            val="-",
            tps="-",
        )

        self.eval_task = self.pbar.add_task(
            "[cyan]eval",
            total=self.cfg.eval.eval_iters,
            visible=False,
            loss="",
            val="",
            tps="",
        )

        self.tokens_per_step = self.cfg.train.batch_size * self.cfg.model.block_size
        self.best_val = float("inf")
        self.ema = None
        self.t_last = time.perf_counter()
        self.t_start = self.t_last

        header = (
            f"{'step':>{2 * len(str(self.training_steps)) + 1}} | {'train':>8} | {'val':>8} | "
            f"{'gap':>7} | {'best':>8}"
        )
        self.console.print(f"[bold cyan]{header}[/]")
        self.console.print(f"[dim]{'─' * len(header)}[/]")

    def log_step(self, loss):
        self.ema = loss if self.ema is None else 0.9 * self.ema + 0.1 * loss
        now = time.perf_counter()
        tps = self.tokens_per_step / (now - self.t_last)
        self.t_last = now
        self.pbar.update(
            self.training_task,
            advance=1,
            loss=f"{self.ema:.3f}",
            tps=f"{tps / 1e3:.1f}k",
        )

    def eval_start(self, split):
        self.pbar.reset(
            self.eval_task,
            total=self.cfg.eval.eval_iters,
            description=f"[cyan]eval {split}",
            visible=True,
        )

    def eval_tick(self):
        self.pbar.update(self.eval_task, advance=1)

    def eval_end(self):
        self.pbar.update(self.eval_task, visible=False)

    def log_eval(self, step, train_loss, val_loss):
        mark = ""
        if val_loss < self.best_val:
            self.best_val, mark = val_loss, " *"
        self.console.print(
            f"[dim]step {step:>5}/{self.training_steps} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | gap {val_loss - train_loss:+.3f} | [dim]best {self.best_val:.4f}{mark}",
        )
        self.pbar.update(self.training_task, val=f"{val_loss:.3f}")

        self.t_last = time.perf_counter()

    def finish_training(self):
        self.pbar.stop()

    def print_summary(self, final_train_loss, final_validation_loss, ckpt_path):
        elapsed = time.perf_counter() - self.t_start

        tokens_seen = self.tokens_per_step * self.training_steps
        delta = final_validation_loss - self.expected_loss
        color = "green" if delta <= 0 else "yellow"

        sections = [
            [
                ("Final train loss", f"{final_train_loss:.4f}"),
                ("Final val loss", f"{final_validation_loss:.4f}"),
                (
                    "Gap (val - train)",
                    f"{final_validation_loss - final_train_loss:+.4f}",
                ),
                ("Best val loss", f"{self.best_val:.4f} @ step {self.best_step}"),
                (
                    "vs expected",
                    f"[{color}]{delta:+.3f}[/] (expected {self.expected_loss:.2f})",
                ),
            ],
            [
                ("Total time", f"{elapsed / 60:.1f} min"),
                ("Tokens seen", f"{tokens_seen / 1e6:.1f} M"),
                ("Avg throughput", f"{tokens_seen / elapsed / 1e3:.1f}k tok/s"),
            ],
            [("Checkpoint", ckpt_path)],
        ]
        blocks = []
        for i, rows in enumerate(sections):
            if i:
                blocks.append(Rule(style="dim green"))  # divider between sections
            grid = Table.grid(padding=(0, 3), expand=True)
            grid.add_column(style="bold green", no_wrap=True)
            grid.add_column(justify="right", overflow="fold")
            for k, v in rows:
                grid.add_row(k, str(v))
            blocks.append(grid)

        self.console.print(
            Panel(
                Group(*blocks),
                title=f"[bold green]RESULTS · {self.cfg.run_name}",
                border_style="green",
                box=box.ROUNDED,
                padding=(1, 2),
                expand=False,
            )
        )
