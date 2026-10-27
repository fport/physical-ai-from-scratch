# Numbers and sources

This book makes one promise: **every number is real.** These are the rules followed while writing it.

## Numbers

1. **A number from our own experiment** is written only from an experiment that was actually run. It comes with the MuJoCo (or library) version, the hardware, and whether the experiment is deterministic.
2. **A number from someone else** (a config line, a document, a paper) comes with its link. Primary sources are preferred: the code itself, official docs, the paper.
3. **The result of an experiment that hasn't been run is never written.** On skeleton pages the experiment stands as a plan; hooks are phrased as questions.
4. Wherever a **success rate** is stated, the number of trials is stated too. From [lesson 22](../phase-4-learning-from-demonstration/22-evaluation.md) on, a confidence interval is given as well.
5. **A claim that can't be verified** is either removed or clearly marked "unverified".

## Simulation and reality

- Simulation results are written as "in simulation". If the real-world counterpart wasn't measured, we don't say "it's the same in reality".
- The output of a model (e.g. MuJoCo's air-drag model) is not presented as a measurement.
- A real robot's training numbers and the numbers used on the real robot are kept apart (see [lesson 11](../phase-2-muscle/11-pd-control.md): 20 / 0.5 in training, 60 / 5 in the real Go2 example).

## Source types

| Mark | Meaning |
|---|---|
| Primary | The code, document or paper itself was read |
| Secondary | Third-party summary, news, search result. Only if no primary source is found, and labelled as such |
| Unverified | The source couldn't be reached, or it doesn't say this. Not used as a claim in the text |

## Copyright

- Robot models come from MuJoCo Menagerie under their own licenses (e.g. `robotstudio_so101` Apache-2.0, `unitree_go2` BSD-3-Clause). Images are our own renders.
- Manufacturer names are used factually; no manufacturer endorses or approves this book.
- No long quotes from other sources; short quotes come with their source.
