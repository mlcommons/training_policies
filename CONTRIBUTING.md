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


## Requirements for an MLPerf Training reference

### General

1. Reference repository code must run *without error* on reference hardware (1xV100) on day of benchmark reference freeze.

2. Compute must be done in *full fp32 precision for any math*.

3. Max runtime is *7 days on 1x V100, fp32*.

    a. An exception from the 7-day @ 1 GPU rule can only come from the Submitter's Working Group.

4. Implementation should be *minimalistic*.

    a. Remove redundant files and features not relevant to the reference

    b. Minimal set of dependencies

    c. Avoid not obvious or hacky solutions (e.g. monkey patching), code should be easy to read 
    and straightforward

5. Command-line arguments:

    a. There should be a command line parameter for every tunable hyperparameter.

    b. Ideally constraints on tunable hparams should be reflected in command line parameter setup (e.g. hyperparameters that must be integers take only integer command line args, not floats) to minimize risk of accidentally running an illegal config.

    c. There can be command line params for non-tunable parameters, but defaults should be set to the correct value.

    d. Hyperparameters may also come from a JSON file, but command-line settings should take precedent over the file, or a warning should be raised.

6. This document applies to ***new*** references.  Existing references should try to adhere as well, but are not required to.

### Hyperparameters & thresholds:

1. There should be an explicit *list of hyperparameters* which can be tuned by submitters, along with tuning rules (e.g. "any positive integer", "grid of allowed values", "one of a few choices" etc.), and allowed optimizers (if more than one).   This should show up in the README and [MLPerf rules doc](https://github.com/mlperf/training_policies/blob/master/training_rules.adoc#91-hyperparameters "hyperparameter rules").

2. The *target accuracy threshold* needs to be explicit in the README and [MLPerf rules doc](https://github.com/mlperf/training_policies/blob/master/training_rules.adoc#3-benchmarks "benchmark table").

### Environment

1. The code should be in a *docker container*, based on the official upstream docker container.

    a. Use the latest public upstream container if you are preparing a new reference model.

2. All *dependencies* should be frozen, with version specified in requirements.txt or in the Dockerfile.

3. Proposal: reference docker image could be uploaded to dockerhub (under mlperf account) to improve reproducibility.

### Implementation features

1. MLPerf-compliant *RNG seeding* must adhere to [RNG rules](https://github.com/mlperf/training_policies/blob/master/training_rules.adoc#51-random-numbers "training rules doc").

2. *Gradient Accumulation* (to emulate large batch training on a few GPUs) basic experiments should be performed to verify that gradient accumulation closely emulates large-batch training.

3. Support for *single-node multi-GPU training* is optional, but encouraged.

    a. Each GPU should get its own process (to reduce overheads) and data parallel is preferred over model parallel (or other techniques).

    b. The reference may support multi-gpu validation, but this step needs to be implemented carefully (e.g. batch norm statistics should be all-reduced across workers to make sure that all replicas are evaluating the same model).

4. Support for *MLPerf logging* is required.

    a. Initial support, at least, should be ready by reference freeze time.  The final list of logged hyperparameters depends on what would be modifiable by submitters.

    b. When the final list of tunable hyperparameters is ready, the final implementation of reference MLPerf logging should be made available.  This likely also require changes to the compliance checker to enforce legal values of hyperparameters.

5. Execution should be *deterministic* if possible, following rules established in the [convergence document](https://docs.google.com/document/d/15DBV5mM8KHYMjGRsJiztQaz-uxKaekOr2pnwmQl_RT0/edit#heading=h.m94pu2k61l60 "google doc").

6. Support for *multi-node training* is optional, but encouraged.  This support does not have to be documented in the public README.

7. Support for *mixed precision training w/ [AMP](https://developer.nvidia.com/automatic-mixed-precision)* is optional, but encouraged.

### Data

1. *Justification* for setting target accuracy must be provided.

    a. Training to target should be reasonably *stable*.  Many random seeds should reach the target with similar number of steps/epochs

    b. Target should be as close to *state-of-the-art* as possible

2. Given a proposed target accuracy on a few (around 10 - 100) random seeds, *all seeds should reach target accuracy*. Steps-to-convergence variance should be as low as possible

3. *Convergence curves* as specified by [Bounded Convergence Document](https://docs.google.com/document/d/15DBV5mM8KHYMjGRsJiztQaz-uxKaekOr2pnwmQl_RT0/edit#heading=h.m94pu2k61l60 "google doc") must be reviewed by the Submitter's Working Group.

### Scripts

1. `run_and_time.sh` script - to execute the benchmark

2. `download_dataset.sh` script - to download dataset and do the initial data preprocessing

3. `verify_dataset.sh` script - to verify correctness of preprocessed data, usually checks md5 sums

4. if training starts from *pretrained checkpoint (or backbone)*:

    a. script to *download* pretrained checkpoint (or backbone)

    b. scripts to *convert* pretrained checkpoint (or backbone) to other popular frameworks should be available

### README

1. brief description of problem, requirements, environment, preprocessing steps, training data, model, optimizer and target metric (description of the metric, target value, evaluation frequency, size of eval dataset).  See [this](https://github.com/mlperf/training_policies/blob/master/training_rules.adoc#3-benchmarks) section from the rules

2. Three summaries are expected.  
    a. Section 1 of the readme should be a very high level description of the task, for a *reader with zero background of machine learning*.  

    b. Following the high level description in section 1 should be a description for *technical press*, who have some machine learning context, so would be interested in more details.  
    
    c. Section 4 should describe the problem to a *machine learning practitioner*, and also include a *link to the paper* describing that network.

