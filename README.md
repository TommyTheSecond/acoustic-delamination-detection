# Acoustic Delamination Detection

A binary classifier that distinguishes intact from delaminated concrete using the sound of a knock.

Concrete delaminates internally when water reaches the rebar and the resulting corrosion expands, opening a
void between the steel and the surface. The defect is invisible from outside. The standard field method for
locating it is the tap test: strike the surface and listen. Intact concrete rings; delaminated concrete
returns a flat, damped thud. Performing the test reliably requires a trained ear, which makes it a
reasonable candidate for automation.

This repository contains the dataset, the feature pipeline, and the model. 580 knock recordings across 37
concrete pillars are reduced to 396-dimensional feature vectors and classified by an ensemble of an RBF-SVM,
a gradient boosting model, and a random forest, with an abstention zone for low-confidence clips.

## Results

| | First model | Current model |
|---|---|---|
| Cross-validation accuracy | 96.3% | 87.1% |
| 6 unseen pillars, per-pillar | 50.0% | 100% (6/6) |
| 6 unseen pillars, per-clip | 56.9% | 91.7% |

The cross-validation figure decreased while performance on unseen pillars roughly doubled. The reason for
that is the substance of this project.

![Holdout accuracy comparison](figures/viz_headline_accuracy.png)

## Why the first model failed

The initial model was an RBF-SVM over 128 mel features. Leave-one-pillar-out cross-validation placed it at
96.3%. Tested afterwards on six newly recorded pillars, three intact and three delaminated, it classified 3
of 6 correctly, and every intact pillar was labelled delaminated.

Four hypotheses were checked:

Overfitting was ruled out. Training accuracy was 91%, below the cross-validation figure, indicating no
memorization of the training set.

Class imbalance was ruled out. Ablating `class_weight` produced no measurable change.

Distribution shift was confirmed. Projecting both sets into a shared PCA basis showed the new intact clips
falling inside the region the model had learned as delaminated. The recordings were made in a different room
at a different microphone distance, which moved the features while the decision boundary stayed fixed.

Biased pillar selection was also confirmed. The training set used 27 of 31 available pillars; four had been
excluded for noisy recording quality. Those four were the only examples of varied recording conditions in
the dataset, so removing them removed the only signal about recording variability the model could have
learned from.

![Feature space PCA](figures/viz_feature_space.png)

The 96.3% was not a false measurement, but it did not measure generalization. Every cross-validation fold
drew its test pillar from the same small set of recording sessions as its training pillars, so holding out a
pillar never held out a recording condition.

## Changes in the current model

**All 31 training pillars, no exclusions.** The four previously excluded pillars were restored.
Cross-validated accuracy fell from 96.3% to 87.1%, reflecting the additional recording variability now
present in the evaluation rather than a degradation of the model.

**Per-clip CMVN.** Normalizing each mel spectrogram to zero mean and unit variance per frequency band
removes the constant channel response of the microphone and room. The technique is standard in speech
recognition, where the equivalent problem is the same speaker recorded over different transmission channels.

**Augmentation ×5.** Each training clip is expanded into five: volume jitter (±6 dB, for source distance),
additive Gaussian noise (20–35 dB SNR, for room noise), onset jitter (±10 ms, for windowing error), and
pitch shift (±0.4 semitones, for resonance variation). Holdout clips are never augmented.

**396 features instead of 128.** The original set was mel mean and standard deviation. Added: CMVN-normalized
mel, delta-mel capturing temporal evolution, eight spectral shape descriptors, three temporal descriptors,
and a decay time τ fitted to the post-onset RMS envelope. τ is the physically motivated feature: intact
concrete dissipates vibrational energy slowly, delaminated concrete dissipates it rapidly.

**Three classifiers rather than one.** An RBF-SVM for smooth nonlinear boundaries, gradient boosting for
feature interactions, and a random forest for noise tolerance, averaged over `P(bad)`. At 31 pillars there is
insufficient data to identify the correct inductive bias in advance, so averaging across three reduces the
cost of choosing wrongly.

**Abstention.** Clips falling in `P(bad) ∈ [0.4, 0.6]` are excluded from the vote, and the pillar verdict is
the majority of the remaining clips. A prediction at 51% carries different information from one at 99% and
is not treated as equivalent.

![Pipeline](figures/viz_pipeline.png)

## Dataset

580 recordings, mono, 16 kHz, each containing one knock of roughly 0.15–0.58 seconds.

```
data/
├── good/      245 clips, 13 pillars    intact concrete
├── bad/       263 clips, 18 pillars    delaminated
├── holdout/    72 clips,  6 pillars    good 14–16, bad 19–21
└── raw/                                unsegmented .m4a source recordings
```

Filenames encode pillar and knock: `goodConcrete3-07.wav` is the 7th knock on good pillar 3. The holdout
subdirectories carry a trailing digit left by the segmentation script that split them out of the `.m4a`
originals, so `bad_194/` is pillar 19 and `good_144/` is pillar 14. The notebooks strip it before reporting.

The six holdout pillars were separated before any of the current model's development and were used once, in
the final evaluation cell. No feature, hyperparameter, model, or threshold was selected with reference to
them.

## Evaluation protocol

Evaluation is leave-one-pillar-out, never a random clip-level split. All clips from a single pillar share a
recording session, a microphone position, and one physical piece of concrete, so a random split places
near-duplicates on both sides of the partition and inflates the result substantially.

![Waveforms](figures/viz_waveforms.png)

![Mel spectrograms](figures/viz_mel_spectrograms.png)

The class difference is partially visible in the spectrograms, with intact knocks retaining low-band energy
longer, but it is not consistent enough across 580 clips to be applied by inspection.

## Detailed results

Six held-out pillars:

| Configuration | Per-pillar | Per-clip |
|---|---|---|
| Ensemble, threshold 0.5 | 83.3% (5/6) | 91.7% (66/72) |
| Ensemble with abstention | 100% (6/6) | 91.7% |
| Gradient boosting alone | 100% (6/6) | 95.8% |
| Random forest alone | 100% (6/6) | 95.8% |
| SVM alone | 83.3% (5/6) | 91.7% |
| First model | 50.0% (3/6) | 56.9% |

Mean `P(bad)` was 13.6% on intact pillars and 82.3% on delaminated ones. 3 of 72 clips abstained, a rate of
4.2%, corresponding to roughly one knock in a five-knock sequence.

![Per-clip predictions](figures/viz_per_clip_with_abstention.png)

Within distribution, leave-one-pillar-out over the 31 training pillars gives 87.1% for the ensemble and
93.5% for the random forest alone.

The two headline figures measure different quantities and are reported together. 87.1% covers 31 pillars
including the noisy recordings; 100% covers six pillars, a sample small enough that a single additional
error would reduce it to 83%. Either figure quoted alone misrepresents the result.

### Failure case: bad_21

`bad_21` is the only pillar the ensemble misclassifies at threshold 0.5. Of its ten clips, five are labelled
intact, giving a mean `P(bad)` of 58% and a tied vote resolved incorrectly. Under abstention, three
borderline clips are removed and the remaining seven split 4–3 toward the correct verdict, which is a narrow
margin. Gradient boosting and the random forest each classify it correctly without assistance; the SVM
accounts for most of the error in the average.

The pillar warrants physical inspection. A plausible explanation is genuinely marginal delamination, such as
an early-stage or thin void, in which case the model is responding to a real property of the sample.

![Confusion matrix](figures/viz_confusion_matrix.png)

## Running the notebooks

Python 3.14 with librosa, scikit-learn, numpy, and matplotlib.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Train and evaluate. The first run takes about ten minutes, most of it feature extraction; subsequent runs
take about five once features are cached to `model/`.

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 notebooks/Ensemble_Abstention_Model.ipynb
```

Regenerate the figures. This reads the feature cache, so the model notebook must be run first.

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 notebooks/Data_Visualization.ipynb
```

| Notebook | Contents |
|---|---|
| `Ensemble_Abstention_Model.ipynb` | Feature extraction, augmentation, LOPO cross-validation, final fit, holdout evaluation. |
| `Ensemble_Abstention_Model_Annotated.ipynb` | The same pipeline with the reasoning for each step, plus a cell for classifying an arbitrary audio file. |
| `Data_Visualization.ipynb` | Generates the nine figures in `figures/`. |

The notebooks locate the project root by walking up from the working directory until `data/` and `model/`
are found, so they run from any location within the repository. Committed outputs come from a full run with
no cache present, and the figures above were generated from that same run.

The classifiers are seeded. The augmentation RNG is seeded once at import rather than per clip, so executing
cells out of order changes which noise draw is applied to which clip. Headline figures are stable across
runs; individual sub-metrics can vary by about a percentage point.

## Limitations

31 pillars is the binding constraint. The relevant sample size is pillars rather than the 508 training
clips, since clips from one pillar are not independent observations. Augmentation instead of additional
model capacity, three sklearn models instead of a neural network, and a six-pillar holdout instead of a
conventional test set are all consequences of that number.

Six holdout pillars is a small test. 6/6 is the strongest available result at this sample size and carries
correspondingly wide error bars.

Recording conditions are uncontrolled. All recordings were taken handheld. A fixed mounting jig would remove
most of the remaining variance and would likely yield more improvement than further modelling.

Binary labels compress a continuous property. Delamination varies in severity, and `bad_21` illustrates the
result of forcing a marginal case into two classes.

Further work, in order of expected value: 50 or more pillars per class recorded under at least three
distinct conditions, a fixed recording jig, and severity as an ordinal label in place of a binary one.

## Repository layout

```
notebooks/     model, annotated walkthrough, figure generation
data/          580 wav clips and raw source recordings
figures/       the nine figures used above
model/         feature cache, generated on first run (gitignored)
```

## License

MIT, see [LICENSE](LICENSE). The audio recordings are covered by the same terms; attribution is appreciated
if the dataset is reused.
