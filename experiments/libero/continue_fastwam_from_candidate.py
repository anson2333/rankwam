#!/usr/bin/env python3
"""Continue FastWAM closed-loop control after a counterfactual candidate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from libero.libero import benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.collect_fastwam_candidates import _model_dimensions, _predict
from experiments.libero.eval_libero_single import (
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
)
from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env
from experiments.libero.task_progress import score_task_progress
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--max-continuation-steps", type=int, default=60)
    args, overrides = parser.parse_known_args()
    if args.max_continuation_steps <= 0:
        raise ValueError("max-continuation-steps must be positive")
    return args, overrides


def main() -> None:
    args, overrides = parse_args()
    with np.load(args.input, allow_pickle=False) as bundle:
        initial_state = np.asarray(bundle["initial_state"]).copy()
        prefix_actions = np.asarray(bundle["prefix_actions"], dtype=np.float32).copy()
        candidate_actions = np.asarray(bundle["candidate_actions"], dtype=np.float32)[0]

    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="sim_libero", overrides=overrides)
    if cfg.ckpt is None:
        raise ValueError("pass ckpt=/path/to/checkpoint as a Hydra override")
    started = time.time()
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    stats_path = _resolve_dataset_stats_path(cfg)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    action_horizon, input_w, input_h = _model_dimensions(cfg)
    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)
    expected = np.asarray(initial_states[args.trial_id])
    if float(np.max(np.abs(initial_state - expected))) > 1e-7:
        raise ValueError("candidate initial state does not match selected trial")

    env_seed = int(cfg.get("seed", 42))
    replan_steps = int(cfg.EVALUATION.replan_steps)
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, env_seed, env_num=1)
    records = []
    policy_step = 0
    done = False
    try:
        np.random.seed(env_seed)
        env.reset()
        obs = env.set_init_state(initial_state.copy())
        for action in prefix_actions:
            obs, _, done, _ = env.step(action)
            if done:
                raise RuntimeError("episode ended while replaying prefix")
        for action in candidate_actions:
            obs, _, done, _ = env.step(action)
            if done:
                break
        if done:
            raise RuntimeError("candidate ended episode before continuation")
        progress = score_task_progress(env, obs, suite=suite_name, task_id=task_id)
        records.append({"continuation_step": 0, "progress": progress})
        while not done and policy_step < args.max_continuation_steps:
            chunk = _predict(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                seed=env_seed,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            execute = min(replan_steps, args.max_continuation_steps - policy_step)
            for action in chunk[:execute]:
                obs, _, done, _ = env.step(action)
                policy_step += 1
                if done:
                    break
            progress = score_task_progress(env, obs, suite=suite_name, task_id=task_id)
            records.append({"continuation_step": policy_step, "progress": progress})
        terminal_state = env.get_sim_state().copy()
        success = bool(env.check_success())
    finally:
        env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "suite": suite_name,
        "task_id": task_id,
        "trial_id": args.trial_id,
        "candidate_action_steps": int(len(candidate_actions)),
        "continuation_steps": policy_step,
        "success": success,
        "done": bool(done),
        "progress": records,
        "terminal_state": terminal_state.tolist(),
        "duration_seconds": time.time() - started,
    }
    (args.output_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("success", "done", "continuation_steps", "duration_seconds")}, indent=2))


if __name__ == "__main__":
    main()
