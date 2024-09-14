# Learn2Clean4Events: Extension for Survival Analysis

This project extends the Learn2Clean framework to allow it to work with survival analysis models. It provides enhanced flexibility and control over the data cleaning process, enabling users to optimize preprocessing pipelines for improved model performance.

## Installation

**Environment:**

*   The project is developed and tested using **conda miniforge3** and **python 3.8.19**. It is highly recommended to use conda for managing the environment, as some packages might be easier to install or have better compatibility with conda.

**Steps:**

1.  **Create a conda environment:**

    ```bash
    conda create -n learn2clean-ext python=3.8.19
    conda activate learn2clean-ext
    ```

2.  **Install dependencies:**

    ```bash
    # Install core dependencies using conda
    conda install numpy=1.23.0 tensorflow=2.11.0 scikit-learn=1.3.2 scipy=1.10.1 Theano=1.0.5 cachetools=5.3.3 scikit-survival=0.22.2 
    impyute=0.0.8 keras=2.11.0 statsmodels=0.14.1 py_stringsimjoin=0.3.5  sklearn-contrib-py-earth=0.1.0 seaborn=0.13.2 jellyfish=1.1.0 tdda=1.0.13 lifelines=0.27.8 tf_slim=1.1.0
    ```

**Note:** If you encounter any issues during installation, please refer to the troubleshooting section below.

## Usage

There are 2 files used to run the framework; **app.py** and **run.py**. They are exactly the same thing except that app.py prompts you to enter values one by one so you will know what you're doing. On the other hand, run.py will allow you to pass arguments right away without having to go through interaction steps or prompts. In case you're using the normal mode (original Learn2Clean package) you will have to specify the dataset path in app.py/run.py.

## Troubleshooting

1: **Try to use both commands "conda install" and "pip install" as pip can sometimes cause issues and conda as well.**

2: **Check that libraries names are not changed or misspelled.**

3: **Check https://anaconda.org for packages as you can download packages from other developers and not necessarily conda:forge.**

3: If the following error is encountered: **"AttributeError: module 'numpy' has no attribute 'ndarray'"**, run "conda install numpy" which will reinstall/update numpy and the issue will be resolved; however, version 1.23.0 works fine.
