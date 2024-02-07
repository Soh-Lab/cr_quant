# Cross-Reactive Quantification

## Why should you care?

Accurate measurement of proteins and small molecules is fundamental to medical diagnostics and
monitoring therapeutic response. Gold standard methods for quantitative measurements often rely on
affinity reagents that exclusively bind to the target molecule to be measured. However, in practice,
many affinity reagents also bind to off-target molecules, which is known as cross-reactivity.
Cross-reactivity can lead to inaccurate quantification through false positives. This work develops a
mathematical framework that corrects for cross-reactivity in measurements and provides a method to
predict the precision of our measurement. This demonstrates the potential to expand the repertoire
of affinity reagents useful for quantification and improve molecular measurement accuracy without
having to change affinity reagents or design new assays.

## Paper

This is the code-base accompanying the
paper "[Theoretical framework and experimental validation of multiplexed analyte quantification using cross-reactive affinity reagents](https://www.biorxiv.org/content/10.1101/2023.11.24.568623v1)",
currently in pre-print on biorxiv.


## Structure of Repository

- ```data```: Data used in the paper with metadata.
- ```explanations```: Code used to generate explanatory plots/animations for the concepts of
  confidence intervals and cross-reactive binding curves.
- ```demos```: Code that implements and demonstrates the methods from the paper without using
  real-world data.
- ```applications```: Code that implements and demonstrates the methods from the paper with
  real-world data.
- ```cr_utils```: Utility functions that implement the lower-level operations described in the
  paper.

## Visual intuition
For an affinity reagent that can bind to two target molecules, because of noise, many different combinations of target molecule concentrations can generate the same signal.
In the following video, the red dot marks the true target molecule concentration mixture, and the region shaded in red marks all the target molecule concentrations that could have generated the same signal readout due to noise.

https://github.com/Soh-Lab/cr_quant/assets/21903833/3298b46c-efa7-4a15-9a46-5a5b5959cdbc
