# IllumiCurveNet evaluation reports

Self-contained HTML reports (open directly in any browser, no server needed —
images are embedded, only the Google Fonts stylesheet needs internet access).
Each one is a snapshot of the model's behavior at a specific point in an
iterative improvement process; later reports don't replace earlier ones —
read them in order to see what changed and why.

| # | Report | What changed | Headline result |
|---|--------|--------------|------------------|
| 1 | [`01_baseline-vs-naive-finetune.html`](01_baseline-vs-naive-finetune.html) | Evaluated the pretrained baseline on the Crescent-Moon test split, then fine-tuned it as-is (unmodified losses, unmodified data) on the training split. | Baseline already generalizes reasonably (NIQE/BRISQUE drop a lot from enhancement alone). Naive fine-tuning cuts training loss 14.6% but **regresses PIQE by 15%** — the exposure loss target and un-degraded training data don't match this domain. |
| 2 | [`02_exposure-mask-and-synthetic-lowlight.html`](02_exposure-mask-and-synthetic-lowlight.html) | Masked the exposure loss to the illuminated disc only (was pulling the whole mostly-black frame toward a brightness target tuned for terrestrial photos), and trained on synthetically darkened+noised versions of the crescent images instead of the well-exposed source renders. | **Backfired at the original loss weights** — PIQE got *worse* than both baseline (+44%) and v1 (+25%). Masking made the exposure loss too easy, freeing the optimizer to trade away texture/spatial fidelity (L_texture +181%). Also didn't improve handling of genuinely noisy input. Directly motivated report 3. |
| 3 | [`03_loss-weight-rebalance.html`](03_loss-weight-rebalance.html) | Rebalanced the six loss weights (raised `L_spa`/`L_texture`, cut `L_TV`/`L_exp`/`L_color`) based on what report 2's per-component logs showed, keeping the masked exposure loss and synthetic-degradation training from report 2. | **This is the model to use.** Beats the baseline, v1, and v2 on every IQA metric (NIQE, BRISQUE, PIQE) on the actual test images — PIQE 19.3 vs. baseline's 20.4. Caveat: still doesn't beat the untouched baseline on genuinely degraded/noisy input — real low-light robustness remains an open problem. |

## Where the underlying numbers live

- Per-epoch training/validation loss: `../logs/training_log_<run_name>.csv`
- Per-image evaluation metrics: `../eval_runs/<run>/per_image_metrics.csv` and `summary_metrics.json`
- Model snapshots: `../snapshots/model-best-<run_name>.pth`
- **Best model overall: `../snapshots/model-best-v3_rebalanced.pth`**
