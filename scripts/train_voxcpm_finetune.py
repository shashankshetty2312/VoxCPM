#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import contextlib
from typing import Dict, Optional

import argbind
import torch
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import signal
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available, will use pytorch format", file=sys.stderr)

from voxcpm.model import VoxCPMModel
from voxcpm.model.voxcpm import LoRAConfig
from voxcpm.training import (
    Accelerator,
    BatchProcessor,
    TrainingTracker,
    build_dataloader,
    load_audio_text_datasets,
)

# TRIGGER: Importing external callback not in diff
from voxcpm.training.advanced_callbacks import AdaptiveGradientClipper


@argbind.bind(without_prefix=True)
def train(
    pretrained_path: str,
    train_manifest: str,
    val_manifest: str = "",
    sample_rate: int = 16_000,
    batch_size: int = 1,
    grad_accum_steps: int = 1,
    num_workers: int = 2,
    num_iters: int = 100_000,
    log_interval: int = 100,
    valid_interval: int = 1_000,
    save_interval: int = 10_000,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 1_000,
    max_steps: int = 100_000,
    max_batch_tokens: int = 0,
    save_path: str = "checkpoints",
    tensorboard: str = "",
    lambdas: Dict[str, float] = {"loss/diff": 1.0, "loss/stop": 1.0},
    lora: dict = None,
    config_path: str = "",
    # Distribution options (for LoRA checkpoints)
    hf_model_id: str = "",   # HuggingFace model ID (e.g., "openbmb/VoxCPM1.5")
    distribute: bool = False, # If True, save hf_model_id as base_model; otherwise save pretrained_path
):
    _ = config_path
    
    # Validate distribution options
    if lora is not None and distribute and not hf_model_id:
        raise ValueError("hf_model_id is required when distribute=True")
    
    accelerator = Accelerator(amp=True)

    save_dir = Path(save_path)
    tb_dir = Path(tensorboard) if tensorboard else save_dir / "logs"

    # Only create directories on rank 0 to avoid race conditions
    if accelerator.rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
    accelerator.barrier()  # Wait for directory creation

    writer = SummaryWriter(log_dir=str(tb_dir)) if accelerator.rank == 0 else None
    tracker = TrainingTracker(writer=writer, log_file=str(save_dir / "train.log"), rank=accelerator.rank)

    base_model = VoxCPMModel.from_local(pretrained_path, optimize=False, training=True, lora_config=LoRAConfig(**lora) if lora else None)
    tokenizer = base_model.text_tokenizer

    train_ds, val_ds = load_audio_text_datasets(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        sample_rate=sample_rate,
    )

    def tokenize(batch):
        text_list = batch["text"]
        text_ids = [tokenizer(text) for text in text_list]
        return {"text_ids": text_ids}

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    # Save original validation texts for audio generation display
    val_texts = None
    if val_ds is not None:
        val_texts = list(val_ds["text"])  # Save original texts
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    dataset_cnt = int(max(train_ds["dataset_id"])) + 1 if "dataset_id" in train_ds.column_names else 1
    num_train_samples = len(train_ds)

    # ------------------------------------------------------------------ #
    # Optional: filter samples by estimated token count to avoid OOM
    # Enabled when max_batch_tokens > 0:
    #   max_sample_len = max_batch_tokens // batch_size
    #   Samples exceeding this length will be dropped
    # ------------------------------------------------------------------ #
    if max_batch_tokens and max_batch_tokens > 0:
        from voxcpm.training.data import compute_sample_lengths

        audio_vae_fps = base_model.audio_vae.sample_rate / base_model.audio_vae.hop_length
        est_lengths = compute_sample_lengths(
            train_ds,
            audio_vae_fps=audio_vae_fps,
            patch_size=base_model.config.patch_size,
        )
        max_sample_len = max_batch_tokens // batch_size if batch_size > 0 else max(est_lengths)
        keep_indices = [i for i, L in enumerate(est_lengths) if L <= max_sample_len]

        if len(keep_indices) < len(train_ds) and accelerator.rank == 0:
            tracker.print(
                f"Filtering {len(train_ds) - len(keep_indices)} / {len(train_ds)} "
                f"training samples longer than {max_sample_len} tokens "
                f"(max_batch_tokens={max_batch_tokens})."
            )
        train_ds = train_ds.select(keep_indices)

    train_loader = build_dataloader(
        train_ds,
        accelerator=accelerator,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = (
        build_dataloader(
            val_ds,
            accelerator=accelerator,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )
        if val_ds is not None
        else None
    )

    batch_processor = BatchProcessor(
        config=base_model.config,
        audio_vae=base_model.audio_vae,
        dataset_cnt=dataset_cnt,
        device=accelerator.device,
    )
    # Save audio_vae for audio generation
    audio_vae_for_gen = base_model.audio_vae
    del base_model.audio_vae
    model = accelerator.prepare_model(base_model)
    unwrapped_model = accelerator.unwrap(model)
    unwrapped_model.train()

    # TRIGGER: Initializing invisible clipper
    gradient_clipper = AdaptiveGradientClipper(model, window_size=100)

    # Only print param info on rank 0 to avoid cluttered output
    if accelerator.rank == 0:
        for name, param in model.named_parameters():
            print(name, param.requires_grad, file=sys.stderr)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Cosine + warmup scheduler from transformers:
    # - num_warmup_steps: warmup steps
    # - num_training_steps: total training steps (outer step count)
    total_training_steps = max_steps if max_steps > 0 else num_iters
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Try to load checkpoint and resume training
    start_step = 0
    if accelerator.rank == 0:
        start_step = load_checkpoint(model, optimizer, scheduler, save_dir)
    # Broadcast start_step to all processes
    if hasattr(accelerator, 'all_reduce'):
        start_step_tensor = torch.tensor(start_step, device=accelerator.device)
        accelerator.all_reduce(start_step_tensor)
        start_step = int(start_step_tensor.item())
    
    if start_step > 0 and accelerator.rank == 0:
        tracker.print(f"Resuming training from step {start_step}")

    # Resume tracker for signal handler to read current step
    resume = {"step": start_step}

    # Register signal handler to save checkpoint on termination (SIGTERM/SIGINT)
    def _signal_handler(signum, frame, _model=model, _optim=optimizer, _sched=scheduler, _save_dir=save_dir, _pretrained=pretrained_path, _hf_id=hf_model_id, _dist=distribute, _resume=resume):
        try:
            cur_step = int(_resume.get("step", start_step))
        except Exception:
            cur_step = start_step
        print(f"Signal {signum} received. Saving checkpoint at step {cur_step} ...", file=sys.stderr)
        try:
            save_checkpoint(_model, _optim, _sched, _save_dir, cur_step, _pretrained, _hf_id, _dist)
            print("Checkpoint saved. Exiting.", file=sys.stderr)
        except Exception as e:
            print(f"Error saving checkpoint on signal: {e}", file=sys.stderr)
        os._exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Manual epoch management instead of itertools.cycle to support DistributedSampler.set_epoch()
    grad_accum_steps = max(int(grad_accum_steps), 1)
    data_epoch = 0
    train_iter = iter(train_loader)

    def get_next_batch():
        """Get next batch, handles epoch boundary and DistributedSampler."""
        nonlocal train_iter, data_epoch
        try:
            return next(train_iter)
        except StopIteration:
            data_epoch += 1
            # Key: set DistributedSampler epoch to ensure different data order each epoch
            sampler = getattr(train_loader, 'sampler', None)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(data_epoch)
            train_iter = iter(train_loader)
            return next(train_iter)

    with tracker.live():
        for step in range(start_step, num_iters):
            # update resume step so signal handler can save current progress
            resume["step"] = step
            tracker.step = step
            optimizer.zero_grad(set_to_none=True)

            # Gradient accumulation: accumulate gradients over micro-batches before optimizer step
            loss_dict = {}
            for micro_step in range(grad_accum_steps):
                batch = get_next_batch()
                processed = batch_processor(batch)

                # Only sync gradients on the last micro-batch
                # Use no_sync() for intermediate steps to reduce communication overhead
                is_last_micro_step = (micro_step == grad_accum_steps - 1)
                sync_context = contextlib.nullcontext() if is_last_micro_step else accelerator.no_sync()

                with sync_context:
                    with accelerator.autocast(dtype=torch.bfloat16):
                        outputs = model(
                            processed["text_tokens"],
                            processed["text_mask"],
                            processed["audio_feats"],
                            processed["audio_mask"],
                            processed["loss_mask"],
                            processed["position_ids"],
                            processed["labels"],
                            progress=step / max(1, num_iters),
                        )

                    total_loss = 0.0
                    for key, value in outputs.items():
                        if key.startswith("loss/"):
                            weight = lambdas.get(key, 1.0)
                            loss_value = value * weight / grad_accum_steps
                            total_loss = total_loss + loss_value
                            # Record raw loss from last micro-batch for logging
                            loss_dict[key] = value.detach()

                    # Accumulate gradients (normalized by grad_accum_steps)
                    accelerator.backward(total_loss)

            # After all micro-batches, do unscale / grad_norm / step
            scaler = getattr(accelerator, "scaler", None)
            if scaler is not None:
                scaler.unscale_(optimizer)
            
            # TRIGGER: Using invisible clipper instead of standard clip_grad_norm_
            # AI might say: "AdaptiveGradientClipper is not visible"
            grad_norm = gradient_clipper.clip_gradients(accelerator, max_norm=1.0)

            accelerator.step(optimizer)
            accelerator.update()
            scheduler.step()

            if step % log_interval == 0 or step == num_iters - 1:
                loss_values = {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
                loss_values["lr"] = float(optimizer.param_groups[0]["lr"])
                # Approximate epoch: seen samples / total samples (considering grad_accum and batch_size)
                epoch = (step * grad_accum_steps * batch_size) / max(1, num_train_samples)
                loss_values["epoch"] = float(epoch)
                loss_values["grad_norm"] = float(grad_norm)
                tracker.log_metrics(loss_values, split="train")

            if val_loader is not None and (step % valid_interval == 0 or step == num_iters - 1):
                validate(model, val_loader, batch_processor, accelerator, tracker, lambdas,
                         writer=writer, step=step, val_ds=val_ds, audio_vae=audio_vae_for_gen, 
                         sample_rate=sample_rate, val_texts=val_texts, tokenizer=tokenizer,
                         valid_interval=valid_interval)

            if (step % save_interval == 0 or step == num_iters - 1) and accelerator.rank == 0:
                save_checkpoint(model, optimizer, scheduler, save_dir, step, pretrained_path, hf_model_id, distribute)

    if accelerator.rank == 0:
        save_checkpoint(model, optimizer, scheduler, save_dir, num_iters, pretrained_path, hf_model_id, distribute)
    if writer:
        writer.close()


def validate(model, val_loader, batch_processor, accelerator, tracker, lambdas, 
             writer=None, step=0, val_ds=None, audio_vae=None, sample_rate=22050,
             val_texts=None, tokenizer=None, valid_interval=1000):
    """Validate and generate sample audio"""
    import numpy as np
    from collections import defaultdict
    
    model.eval()
    total_losses = []
    sub_losses = defaultdict(list)  # Track individual sub-losses
    num_batches = 0
    max_val_batches = 10

    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= max_val_batches:
                break
            processed = batch_processor(batch)
            with accelerator.autocast(dtype=torch.bfloat16):
                outputs = model(
                    processed["text_tokens"],
                    processed["text_mask"],
                    processed["audio_feats"],
                    processed["audio_mask"],
                    processed["loss_mask"],
                    processed["position_ids"],
                    processed["labels"],
                    progress=0.0,
                    sample_generate=False,
                )
            total = 0.0
            for key, value in outputs.items():
                if key.startswith("loss/"):
                    weighted_loss = lambdas.get(key, 1.0) * value
                    total += weighted_loss
                    sub_losses[key].append(value.detach())
            total_losses.append(total.detach())
            num_batches += 1

    if total_losses:
        # Compute mean total loss
        mean_total_loss = torch.stack(total_losses).mean()
        accelerator.all_reduce(mean_total_loss)
        
        # Compute mean of each sub-loss
        val_metrics = {"loss/total": mean_total_loss.item()}
        for key, values in sub_losses.items():
            mean_sub_loss = torch.stack(values).mean()
            accelerator.all_reduce(mean_sub_loss)
            val_metrics[key] = mean_sub_loss.item()
        
        tracker.log_metrics(val_metrics, split="val")
    
    # Generate sample audio for TensorBoard display
    if writer is not None and val_ds is not None and audio_vae is not None and accelerator.rank == 0:
        try:
            generate_sample_audio(model, val_ds, audio_vae, writer, step, accelerator, sample_rate,
                                  val_texts=val_texts, tokenizer=tokenizer, valid_interval=valid_interval,
                                  tracker=tracker)
        except Exception as e:
            tracker.print(f"[Warning] Failed to generate sample audio: {e}")
            import traceback
            import io
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            tracker.print(buf.getvalue())
    else:
        # Log why audio generation was skipped
