# RankWAM Experiment Protocol

## Scope

The first milestone tests whether FastWAM action candidates have a useful
counterfactual ranking signal in LIBERO. It does not train an action-conditioned
video model or an RL actor.

## Immutable baseline

- FastWAM upstream commit: `45d8e1458921d83f8ad6cf9ce993d371208dabd0`
- LIBERO-plus commit: `4976dc30028e805ff8094b55501d532c48fec182`
- MuJoCo: `3.3.2`
- Pilot suite: `libero_object`
- Pilot task: `0` (`pick up the alphabet soup and place it in the basket`)
- Policy checkpoint: FastWAM release `libero_uncond_2cam224.pt`

Do not modify the default FastWAM model or evaluation configs for pilot work.
RankWAM behavior must be enabled through separate scripts or config files.

## Reproducibility contract

Every run directory must contain:

- resolved config;
- git commit and dirty status for RankWAM and LIBERO-plus;
- Python, PyTorch, CUDA driver, MuJoCo, robosuite and LIBERO paths;
- SHA-256 hashes for checkpoint and dataset statistics;
- environment seed, policy seed and candidate seed;
- raw per-episode and per-candidate records;
- wall time, peak GPU memory and model-call count.

Dataset splits are grouped by source episode. States or candidates from one
source episode may not cross train, validation and test splits.

## Gates

### G0: baseline

Run 5 smoke trials and then 50 trials on the pilot task. Stop if preprocessing,
normalization, or checkpoint loading does not match the upstream evaluation path.

### G1: deterministic isolated replay

Do not branch by restoring only MuJoCo's flattened state. Robosuite controller
goals, observable buffers, timers, and renderer state are not included in that
array. The correctness-first branch protocol is:

1. seed NumPy before construction, then create a fresh environment with the
   recorded environment seed;
2. reset and apply the recorded LIBERO initial state;
3. replay the complete recorded action prefix to the branch point;
4. verify the reconstructed branch state;
5. execute one candidate suffix and destroy the environment.

For at least 100 isolated prefix-replay checks:

- reconstructed branch states must match within the configured tolerance;
- simulator terminal states must match exactly;
- success flags must match exactly;
- proprio observations must match within the configured absolute tolerance;

RGB is a separate rendering audit, not a label-consistency gate. Independent
EGL contexts can produce different edge pixels for identical simulator states.
Report maximum and mean RGB differences and the changed-pixel fraction. Labels
must use simulator state, task success, and task-specific progress only. They
must never use cross-context pixel equality.

Do not collect counterfactual labels until this gate passes.

Candidate collection must store the initial-state index, environment seed, and
the complete executed action prefix. Process-level snapshotting may be added as
an optimization only after it is proven equivalent to isolated prefix replay.

LeRobot demonstration actions are not an expert anchor until their source
simulator state is matched to the target branch. Matching task ids or assuming
`episode_id == initial_state_id` is insufficient: that assumption failed the
task-0 replay probe on the 4051 dataset. Keep demo actions out of the candidate
set until the original demonstration initial state (or an equivalent state
mapping validated by replay success) is available.

### G2: candidate headroom

Collect 50-100 pilot states with `K=8` candidates. Before training a ranker,
report:

- candidate action diversity;
- informative-group rate (both positive and negative outcomes in one group);
- random candidate success;
- first-candidate success;
- simulator-oracle success and uplift.

For the task-0 pilot, also report the versioned `pick_place_v1` progress metric
and every raw component used to compute it. Let `d_reach` be end-effector to
alphabet-soup distance and `d_target` be alphabet-soup to basket-region distance:

- `reach = 1 - tanh(10 * d_reach)`;
- `transport = 1 - tanh(5 * d_target)`;
- before grasp: `progress = 0.49 * reach`;
- while grasped: `progress = 0.5 + 0.49 * transport`;
- on task success: `progress = 1.0`.

Store `grasped`, `in_target`, both distances, and all three positions with the
score. This dense metric diagnoses candidate generation and can define pairwise
preferences, but it does not replace the binary success headroom gate below.

Do not train a ranker if informative-group rate is below 20% or simulator-oracle
uplift is below 10 percentage points. Change candidate generation or state
sampling first.

### G3: offline ranker

Proceed to closed-loop reranking only if held-out pairwise accuracy exceeds 70%,
top-1 regret is significantly below random selection, and three training seeds
agree in direction.

## Statistical unit

The independent unit is a source episode or restored state, never an individual
pair. Confidence intervals use episode/state clustered bootstrap. Pair expansion
does not increase the reported sample count.
