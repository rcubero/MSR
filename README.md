# Multi-Scale Relevance (MSR)

================================

Multi-scale relevance is a method for identifying neurons relevant to the animal's behaviour being probed in an experiment without an a priori knowledge of any external features. It is described in the paper:
> RJ Cubero, M Marsili, Y Roudi<br>
> Finding informative neurons in the brain using Multi-Scale Relevance<br>

This repository contains Python codes for implementing multi-scale relevance and for reproducing the figures in the paper.
-**relevance.py** is the main Python code which is used to calculate for the multiscale relevance. In particular, the function **parallelized_relevance** takes as an input the **total number of time points** (must be the same number as the size of the spike train) and the **spike train** of a single neuron and returns as an output a scalar value which is the **multiscale relevance**.


