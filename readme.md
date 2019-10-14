# A computational account of threat-related attentional bias

Toby Wise, Jochen Michely, Peter Dayan & Raymond J Dolan

_PLoS Computational Biology_, 2019

[https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007341](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007341)

## Code

Analyses for this project are contained within Jupyter notebooks (in the `/notebooks` directory). Python 2.7 was used for all these analysese, and the code will most likely not run smoothly with Python 3.

These notebooks make use of scripts located in the `/code` directory and standard Python packages (e.g. numpy, pandas, matplotlib etc).

In addition, they require a package called [DMPy](https://github.com/tobywise/DMpy/tree/baf71241a1ecff20a3908c99ec236e7a06c49474) (this package is half-written and doesn't have much functionality beyond that required for this project). Eye-tracking analysis is mostly done using a fork of [PyeParse](https://github.com/tobywise/pyeparse) that allows import of iohub eyetracking files.

## Notebooks

The majority of the analysis reported in the paper is run in a series of Jupyter notebooks, which are located in the `/notebooks` directory. The only exception is the fitting of behavioural models, which was run on a HPC cluster for speed (code for this model fitting is provided in the `/code` directory).

There are 5 notebooks, each of which runs a specific section of the analysis pipeline and produces all the figures etc. associated with it.

### Behavioural modelling

This notebook prepares raw behavioural data and constructs computatioanl models. It also runs analysese of the results of model fitting.

### Fixation analysis

This notebook prepares eye tracking data and extracts fixations. It then runs analyses exploring effects of model-derived measures on attention.

### Attention effects on learning

This notebooks performs analyses looking at how attention affects learning.

### Questionnaire analyses

This notebook runs regression analyses examining relationships between questionnaire measures of state/trait anxiety and behavioural variables.

### Parameter recovery

This runs some quick parameter recovery checks for the best fitting models.

## Data

All data is available on the Open Science Framework [here](https://osf.io/b4e72/).

Eye tracking data is compressed as `.gz` files and so will need to be extracted before use. The analysis code expects to find all the data in a directory called `/data`.

Processed data is also provided, including fitted models simulated data from these models.

Data is available for 63 subjects - two subjects were excluded prior to analyis as they started the task but did not complete it.
