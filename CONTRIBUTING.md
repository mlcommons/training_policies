# Contributing guidelines

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Read [contributing guidelines](CONTRIBUTING.md)
- Ensure you have signed the [Contributor License Agreement (CLA)](https://cla.developers.google.com/).
- (Note: additional technical details TBD by community.)

## How to become a contributor and submit your own code

### Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump a couple of legal hurdles.

Please fill out either the individual or corporate Contributor License Agreement (CLA).

  * If you are an individual writing original source code and you're sure you own the intellectual property, then you'll need to sign an [individual CLA](https://code.google.com/legal/individual-cla-v1.0.html).
  * If you work for a company that wants to allow you to contribute your work, then you'll need to sign a [corporate CLA](https://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and instructions for how to sign and return it. Once we receive it, we'll be able to accept your pull requests.

***NOTE***: Only original source code from you and other people that have signed the CLA can be accepted into the main repository. (Note: we need to modify this to allow third party code under Apache2 or MIT license with additional review.)

### Contributing code

If you have improvements to MLPerf, send us your pull requests! For those
just getting started, Github has a [howto](https://help.github.com/articles/using-pull-requests/).

### Contribution guidelines and standards

(Note: Technical details TBD by community.)

#### General guidelines and philosophy for contribution

(Note: Technical details TBD by community.)

#### License

Include a license at the top of new files.

* [C/C++ license example](https://github.com/mlperf/policies/blob/master/license_example_DRAFT.cpp)
* [Python license example](https://github.com/mlperf/policies/blob/master/license_example_DRAFT.py)

#### C++ coding style

(Note: Technical details TBD by community.)

#### Python coding style

(Note: Technical details TBD by community.)

#### Running sanity check

(Note: Technical details TBD by community.)

#### Running unit tests

(Note: Technical details TBD by community.)

## Requirements for a good MLPerf Training reference (work in progress)

### General
1. Can be run without error on reference hardware (1xV100) on day of freeze
2. Compute must be done in full fp32 precision everywhere
3. Max runtime is 7 days on 1x V100, fp32
    a. exception from 7-day @ 1 GPU rule is possible if absolutely necessary
4. Implementation should be minimalistic
remove redundant files and features not relevant to the reference
minimal set of dependencies
avoid not obvious or hacky solutions (e.g. monkey patching), code should be easy to read and straightforward
command-line arguments:
there should be command line param for every tunable hyperparameter
ideally constraints on tunable hparams should be reflected in command line params setup (e.g. hparams that must be integers take only integer command line args, not floats) to minimize risk of accidentally running illegal config
there can be command line params for non-tunable parameters, but defaults should be set to the correct value
This document applies to new references.  Existing references should try to adhere as well, but are not required to.
Hyperparameters & thresholds:
list of hyperparameters which can be tuned by submitters, and tuning rules (e.g. "any positive integer", "grid of allowed values", "one of a few choices" etc.), also allowed optimizers (if more than one)
target accuracy threshold
eventually list of tunable parameters, accuracy threshold etc. should end up in the MLPerf rules doc
Environment
docker container, based on official upstream docker container, 
ideally use the latest public upstream container if you're preparing a new reference model
all dependencies should be frozen 
version specified in requirements.txt or in Dockerfile
proposal: reference docker image could be uploaded to dockerhub (under mlperf account) to improve reproducibility
Implementation features
MLPerf-compliant RNG seeding (rules)
gradient accumulation (to emulate large batch training on a few GPUs)
basic experiments should be performed to verify that gradient accumulation closely emulates large-batch training
support for single-node multi-GPU training
each GPU should get its own process (to reduce overheads) and data parallel is preferred over model parallel (or other techniques)
reference may support multi-gpu validation, but this step needs to be implemented carefully (e.g. batch norm statistics should be all-reduced across workers to make sure that all replicas are evaluating the same model)
support for MLPerf logging
at least initial support, final list of logged hyperparameters depends on what would be modifiable by submitters
when the final list of tunable hyperparameters is ready:
final implementation of reference MLPerf logging
changes to compliance checker to enforce legal values of hyperparameters
deterministic execution (if possible)
Optional support for multi-node training 
should exist in code, doesn't have to be documented in public README
Optional support for mixed precision training w/ AMP
Data
justification for setting target accuracy:
training to target should be reasonably stable, many random seeds should reach the target with similar number of steps/epochs
target should be as close to SOTA as possible
results from a stability tests with reference hyperparameters and proposed target accuracy on a few (~10 - ~100) random seeds, all seeds should reach target accuracy, steps-to-convergence variance should be as low as possible
Convergence curves as specified by Bounded Convergence Document
Scripts
run_and_time.sh script - to execute the benchmark
download_dataset.sh script - to download dataset and do the initial data preprocessing
verify_dataset.sh script - to verify correctness of preprocessed data, usually checks md5 sums
if training starts from pretrained checkpoint (or backbone):
script to download pretrained checkpoint (or backbone)
scripts to convert pretrained checkpoint (or backbone) to other popular frameworks should be available
README
brief description of problem, requirements, environment, preprocessing steps, training data, model, optimizer and target metric (description of the metric, target value, evaluation frequency, size of eval dataset)
see this section from the rules

