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
from rich.text import Text

from utils import MFU


class TrainingDisplay:
    def __init__(
        self,
        cfg,
        vocab_size,
        params,
        plan,
        gpu,
        training_tokens,
        validation_tokens,
        training_steps,
    ):
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.params = params
        self.plan = plan
        self.gpu = gpu
        self.training_tokens = training_tokens
        self.validation_tokens = validation_tokens
        self.training_steps = training_steps
        self.expected_training_time = plan["hours"]
        self.expected_loss = plan["loss"]
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
            title_style="bold #17A005",
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
            ("Embedding params", f"{self.params['embedding'] / 1e6:.2f} M"),
            ("Head params", f"{self.params['head'] / 1e6:.2f} M"),
            ("Non-embedding params", f"{self.params['non_embedding'] / 1e6:.2f} M"),
            ("Total params (N)", f"{self.params['total'] / 1e6:.2f} M"),
            ("Tokens / param", f"{self.plan['tokens_per_param']:.2f}"),
        ]

        for k, v in model_rows:
            model_table.add_row(k, str(v))

        calculated_eval_interval = max(1, self.training_steps // 20)

        training_table = Table(
            title="TRAINING CONFIGURATION",
            title_style="bold #17A005",
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
                "GPU / dtype",
                f"{self.gpu['name'].replace('NVIDIA ', '').replace('GeForce ', '')[:22]} / {self.gpu['dtype']}",
            ),
            (
                "Peak (assumed MFU)",
                f"{self.gpu['peak_flops'] / 1e12:.1f} TF ({MFU:.0%})"
                if self.gpu["peak_flops"]
                else "unknown",
            ),
            (
                "Expected loss / hours",
                f"{self.expected_loss:.2f} / "
                + (
                    f"{self.expected_training_time:.2f}"
                    if self.expected_training_time is not None
                    else "n/a"
                ),
            ),
            (
                "Optimal N / loss",
                f"{self.plan['optimal_params'] / 1e6:.1f} M / {self.plan['optimal_loss']:.2f}",
            ),
            (
                "Optimal N (data)",
                f"{self.plan['optimal_params_for_data'] / 1e6:.1f} M",
            ),
            (
                "Tokens vs 20/param",
                "too few" if self.plan["data_limited"] else "enough",
            ),
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

        # widths match the rows printed in log_eval
        w = len(str(self.training_steps))
        step_w = len(f"step {self.training_steps:>{w}}/{self.training_steps}")
        header = (
            f"{'step':<{step_w}} | {'train':<12} | {'val':<10} | "
            f"{'gap':<10} | {'best':<13}"
        )
        self.console.print()
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
            description=f"[cyan]Estimating {split}",
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
            f"step {step:>{len(str(self.training_steps))}}/{self.training_steps} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | gap {val_loss - train_loss:+.3f} | best {self.best_val:.4f}{mark}",
        )
        self.pbar.update(self.training_task, val=f"{val_loss:.3f}")

        self.t_last = time.perf_counter()

    def finish_training(self):
        self.pbar.stop()
        self.console.print()

    def print_summary(self, final_train_loss, final_validation_loss, ckpt_path):
        elapsed = time.perf_counter() - self.t_start

        tokens_seen = self.tokens_per_step * self.training_steps
        peak = self.gpu["peak_flops"]
        measured_mfu = (
            f"{6 * self.params['total'] * tokens_seen / elapsed / peak:.1%}"
            if peak
            else "n/a"
        )
        delta = final_validation_loss - self.expected_loss
        delta_color = "green" if delta <= 0 else "yellow"
        gap = final_validation_loss - final_train_loss
        gap_color = (
            "green" if abs(gap) < 0.05 else "yellow" if abs(gap) < 0.15 else "red"
        )

        # (label color, value color, rows)
        sections = [
            (
                "cyan",
                "white",
                [
                    ("Final train loss", f"{final_train_loss:.4f}"),
                    ("Final val loss", f"{final_validation_loss:.4f}"),
                    ("Gap (val - train)", f"[{gap_color}]{gap:+.4f}[/]"),
                    ("Best val loss", f"[bold green]{self.best_val:.4f}[/]"),
                    (
                        "vs expected",
                        f"[{delta_color}]{delta:+.3f}[/] (expected {self.expected_loss:.2f})",
                    ),
                ],
            ),
            (
                "magenta",
                "white",
                [
                    ("Total time", f"{elapsed / 60:.1f} min"),
                    ("Tokens seen", f"{tokens_seen / 1e6:.1f} M"),
                    ("Avg throughput", f"{tokens_seen / elapsed / 1e3:.1f}k tok/s"),
                    ("Measured MFU", measured_mfu),
                ],
            ),
            ("yellow", "#f5dfbb", [("Checkpoint", ckpt_path)]),
        ]
        blocks = []
        for i, (label_color, value_color, rows) in enumerate(sections):
            if i:
                blocks.append(
                    Rule(style=f"dim {label_color}")
                )  # divider between sections
            grid = Table.grid(padding=(0, 3), expand=True)
            grid.add_column(style=f"bold {label_color}", no_wrap=True)
            grid.add_column(justify="right", overflow="fold", style=value_color)
            for k, v in rows:
                grid.add_row(k, str(v))
            blocks.append(grid)

        self.console.print()
        self.console.print(
            Panel(
                Group(*blocks),
                title=f"[bold cyan]RESULTS[/] · [bold magenta]{self.cfg.run_name}",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
                expand=False,
            )
        )

    def print_generation(self, text):
        self.console.print()
        self.console.print(
            Panel(
                Text(text),
                title="[bold magenta]GENERATED SAMPLE",
                border_style="magenta",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        self.console.print()
