# Acoustic Delamination Detection

Concrete delaminates from the inside. Water gets into the rebar, the rebar rusts, the rust expands, and a
void opens up between the steel and the surface — all of it invisible from outside. The standard way to find
it is the **tap test**: hit the concrete with a hammer or a coin and listen. Solid concrete rings. Delaminated
concrete gives back a flat, dead thud.

That works, but it takes a trained ear, and a junior engineer on their first survey does not have one yet.
This project trains a classifier to make that call from a phone recording of the knock.

**580 knock recordings across 37 concrete pillars → mel-spectrogram features → an SVM + gradient boosting +
random forest ensemble that abstains when it isn't sure.** Everything runs on a laptop CPU in about ten
minutes. No GPU, no cloud, no deep learning.

The eventual target was a phone doing this offline in the field, which is why the signal-processing constants
are fixed at 16 kHz and a 200 ms window and why the classifiers are small enough to serialize. This repo is
the model and the dataset; the mobile side isn't part of it.

| | Original model | This model |
|---|---|---|
| Cross-validation accuracy | 96.3% | 87.1% |
| **6 genuinely unseen pillars** | **50.0%** | **100% (6/6)** |
| Per-clip on those pillars | 56.9% | 91.7% |

Those two columns are the whole story, and the interesting part is that the first row got *worse*.

![Holdout accuracy comparison](figures/viz_headline_accuracy.png)

---

## The 96% that wasn't real

The first version of this was a straightforward RBF-SVM on 128 mel features. Leave-one-pillar-out
cross-validation put it at 96.3%, which felt like a finished project.

Then I recorded six new pillars — three good, three delaminated — with the same phone, on a different day.
The model got 3 out of 6. Every single good pillar was called delaminated. On the pillars it had never seen,
it was worse than a coin flip that always guesses "bad."

Diagnosing it ruled things out one at a time:

- **Not overfitting.** Training accuracy was 91%, below the CV number. There was no memorization to find.
- **Not class imbalance.** Ablating `class_weight` changed nothing.
- **It was distribution shift.** A PCA projection showed the new *good* clips landing squarely inside the
  region of feature space the model had learned as *bad*. New room, new mic distance, new background noise —
  the features moved, and the decision boundary didn't follow.
- **And the pillar selection was rigged, by me.** The original training set used 27 of the 31 pillars. Four
  had been dropped as "noisy recordings." Those four were the only examples of varied recording conditions
  in the entire dataset. I had thrown away exactly the data that would have taught the model to handle a
  new room.

![Feature space PCA](figures/viz_feature_space.png)

The 96% was measuring the wrong thing. Every fold of that cross-validation drew its test pillar from the
same handful of recording sessions as its training pillars, so "unseen pillar" never meant "unseen
conditions." The number was real; what it implied was not.

## What changed

Six things, roughly in order of how much they mattered:

**All 31 training pillars, no exclusions.** The noisy ones went back in. Cross-validated accuracy immediately
fell from 96.3% to 87.1%. That drop is the point — the model didn't get worse, the measurement got honest.

**Per-clip CMVN.** Normalizing each mel spectrogram to zero mean and unit variance per frequency band strips
out the constant channel response of the mic and the room. It's borrowed straight from speech recognition,
where the same problem shows up as "same speaker, different telephone."

**Augmentation ×5.** Each training clip becomes five: volume jitter (±6 dB, standing in for mic distance),
Gaussian noise at 20–35 dB SNR (different rooms), onset jitter (±10 ms, sloppy windowing), and pitch shift
(±0.4 semitones, slight resonance variation). Holdout clips are never augmented.

**396 features instead of 128.** The originals were mel mean and standard deviation. Added: CMVN'd mel,
delta-mel for how the sound evolves, eight spectral shape descriptors, three temporal ones, and a decay time
τ fitted to the post-onset RMS envelope. That last one is the physically motivated one — intact concrete
dissipates vibrational energy slowly, delaminated concrete kills it fast — and it's the feature I'd keep if
I had to keep only one.

**Three models instead of one.** RBF-SVM for smooth nonlinear boundaries, gradient boosting for feature
interactions, random forest for noise tolerance. Average their `P(bad)`. With 31 pillars there isn't enough
data to know in advance which inductive bias is right, so the ensemble hedges.

**Abstention.** Any clip landing in `P(bad) ∈ [0.4, 0.6]` doesn't get a vote. The pillar verdict is the
majority of the clips that were actually confident. A 51% prediction and a 99% prediction are not the same
claim and shouldn't be counted the same way.

![Pipeline](figures/viz_pipeline.png)

## The data

580 recordings, all mono 16 kHz, each one knock lasting roughly 0.15–0.58 seconds.

```
data/
├── good/      245 clips, 13 pillars    intact concrete
├── bad/       263 clips, 18 pillars    delaminated
├── holdout/    72 clips,  6 pillars    good 14–16, bad 19–21
└── raw/                                unsegmented .m4a source recordings
```

Filenames encode pillar and knock: `goodConcrete3-07.wav` is the 7th knock on good pillar 3. The holdout
subdirectories carry a trailing digit from the segmentation script that split them out of the `.m4a`
originals — `bad_194/` is pillar 19, `good_144/` is pillar 14. The notebooks strip it before reporting.

The six holdout pillars were locked away before any of the v1 work started and touched exactly once, in the
final evaluation cell. Nothing about the feature design, the model choice, the hyperparameters, or the
abstention window was chosen with any knowledge of them. That discipline is the only reason the 100% means
anything.

**Evaluation is leave-one-pillar-out, never a random clip split.** All the clips from one pillar share a
recording session, a microphone position, and a physical piece of concrete. Splitting them randomly puts
near-duplicates on both sides of the line and produces a number in the high nineties that means nothing.

![Waveforms](figures/viz_waveforms.png)

![Mel spectrograms](figures/viz_mel_spectrograms.png)

The difference is visible if you know where to look — good knocks hold energy in the low bands longer — but
it's not something you'd reliably eyeball across 580 clips, which is the argument for a classifier.

## Results

On the six held-out pillars:

| Setup | Per-pillar | Per-clip |
|---|---|---|
| Ensemble, threshold 0.5 | 83.3% (5/6) | 91.7% (66/72) |
| **Ensemble + abstention** | **100% (6/6)** | 91.7% |
| Gradient boosting alone | 100% (6/6) | 95.8% |
| Random forest alone | 100% (6/6) | 95.8% |
| SVM alone | 83.3% (5/6) | 91.7% |
| Original SVM | 50.0% (3/6) | 56.9% |

Mean `P(bad)` was 13.6% on the good pillars and 82.3% on the delaminated ones. 3 clips out of 72 abstained —
4.2%, or about one knock in a five-knock session, where the fix is "knock again."

![Per-clip predictions](figures/viz_per_clip_with_abstention.png)

Within-distribution, leave-one-pillar-out over the 31 training pillars gives 87.1% for the ensemble and
93.5% for the random forest alone.

**Report both numbers or neither.** 87.1% cross-validated and 100% on holdout look contradictory until you
notice they measure different things: 87.1% is 31 pillars' worth of hard cases including the noisy
recordings, and 100% is six pillars — a sample small enough that one bad pillar would have made it 83%.
Quoting either alone is misleading in a different direction.

### The one that nearly broke it

`bad_21` is the only pillar the ensemble gets wrong at threshold 0.5. Ten clips, five of them called good,
mean `P(bad)` of 58% — a tied vote resolved the wrong way. With abstention, three borderline clips drop out
and the remaining seven split 4–3 in favour of BAD, which is correct but not exactly convincing. Gradient
boosting and random forest each get it right on their own; the SVM is what drags the average down.

I'd want to physically inspect that pillar. My guess is it's genuinely marginal — early-stage delamination,
or a thin void — and that the model is picking up something real.

![Confusion matrix](figures/viz_confusion_matrix.png)

## Running it

Python 3.14 with librosa, scikit-learn, numpy and matplotlib. Developed on macOS, nothing platform-specific.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Train and evaluate — around 10 minutes the first time, most of it feature extraction, then ~5 minutes on
re-runs once features are cached to `model/`:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 notebooks/Ensemble_Abstention_Model.ipynb
```

Regenerate the figures (~30 seconds, but it reads the feature cache, so run the model notebook first):

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 notebooks/Data_Visualization.ipynb
```

Or open them and step through:

```bash
.venv/bin/jupyter notebook notebooks/Ensemble_Abstention_Model_Annotated.ipynb
```

| Notebook | What it is |
|---|---|
| `Ensemble_Abstention_Model.ipynb` | The model. Feature extraction, augmentation, LOPO, final fit, holdout evaluation. |
| `Ensemble_Abstention_Model_Annotated.ipynb` | Same pipeline, written out with the reasoning at each step, plus a cell for classifying a single new recording. |
| `Data_Visualization.ipynb` | Produces the nine figures in `figures/`. |

The notebooks find the project root by walking up from the working directory until they see `data/` and
`model/`, so they run from anywhere in the repo. Committed outputs are from a real end-to-end run with no
cache present — the numbers in this README can be checked against them without executing anything.

One caveat on exact reproducibility: the classifiers are seeded, but the augmentation RNG is seeded once at
import rather than per clip, so running the cells out of order shifts which noise draw lands on which clip.
Headline figures come out identical run to run; individual sub-metrics can move by a point or so.

## Limits

**N=31 pillars is the binding constraint on everything.** Not 508 clips — 31 pillars, because clips from one
pillar are not independent observations. Every design decision here is downstream of that number: it's why
there's augmentation instead of more capacity, why it's three sklearn models instead of a CNN, and why the
holdout is six pillars instead of a proper test set.

**Six holdout pillars is a small test.** 6/6 is the best result available at this sample size and it still
has wide error bars. It's evidence the rebuild worked, not proof the model is 100% accurate.

**Recording conditions still aren't controlled.** Everything was recorded handheld. A fixed phone-to-surface
jig would remove most of the remaining variance, and would probably do more for accuracy than any further
modelling.

**Binary labels flatten a spectrum.** Delamination has degrees, and `bad_21` is what that looks like when
you force it into two classes.

What I'd do next, in order: 50+ pillars per class recorded under at least three different conditions, a
recording jig, then severity as an ordinal label rather than a binary one.

## Layout

```
notebooks/     model, annotated walkthrough, figure generation
data/          580 wav clips + raw source recordings
figures/       the nine figures used above
model/         feature cache, generated on first run (gitignored)
```

## License

MIT — see [LICENSE](LICENSE). The audio recordings are included under the same terms; attribution is
appreciated if you use the dataset.
