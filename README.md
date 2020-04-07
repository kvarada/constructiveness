### Constructiveness in online news comments
==============================

Identifying constructive language

News organizations enable online commenting with the goal of increasing user engagement and enhancing the quality of news and information. Unfortunately, online commenting platforms are often used as a place to vent anger, frustration, and express hatred, in general or towards specific groups. While there is a clear value in including reader comments, news organizations find themselves having to sift through and organize comments to ensure that commenters exchange views, opinions, and insights in a respectful and constructive manner, and to keep the discussion relevant to the issues discussed in the article. The goal of this project is to promote meaningful and constructive discussion online. Accordingly, we have developed computational methods to identify constructive comments. 

The data for this project is a subset of comments from the SFU Opinion and Comments Corpus. This subset, the Constructive Comments Corpus (C3) consists of 12,000 comments annotated by crowdworkers for constructiveness and its characteristics.

- [SFU Opinion and Comments Corpus](https://github.com/sfu-discourse-lab/SOCC)
- Constructive Comments Corpus (C3)
    - On Kaggle: [https://www.kaggle.com/mtaboada/c3-constructive-comments-corpus](https://www.kaggle.com/mtaboada/c3-constructive-comments-corpus)
    - On Simon Fraser University's Repository: [https://researchdata.sfu.ca/islandora/object/sfu%3A2977](https://researchdata.sfu.ca/islandora/object/sfu%3A2977)
    - Citation: Kolhatkar, V., N. Thain, J. Sorensen, L. Dixon and M. Taboada (2020) C3: The Constructive Comments Corpus. Jigsaw and Simon Fraser University. (Dataset). DOI: 10.25314/ea49062a-5cf6-4403-9918-539e15fd7b52

For more information about this work, please see our papers. 

- Kolhatkar, V., N. Thain, J. Sorensen, L. Dixon and M. Taboada (to appear) Classifying constructive comments. Journal article under review.  

- Kolhatkar, V.,H. Wu, L. Cavasso, E. Francis, K. Shukla and M. Taboada (to appear) The SFU Opinion and Comments Corpus: A corpus for the analysis of online news comments. Corpus Pragmatics.

- Kolhatkar. V. and M. Taboada (2017) [Using New York Times Picks to identify constructive comments](https://www.aclweb.org/anthology/W17-4218/). [Proceedings of the Workshop Natural Language Processing Meets Journalism](http://nlpj2017.fbk.eu/), Conference on Empirical Methods in Natural Language Processing. Copenhagen. September 2017.

- Kolhatkar, V. and M. Taboada (2017) [Constructive language in news comments](http://aclweb.org/anthology/W17-3002). [Proceedings of the 1st Abusive Language Online Workshop](https://sites.google.com/site/abusivelanguageworkshop2017/), 55th Annual Meeting of the Association for Computational Linguistics. Vancouver. August 2017, pp. 11-17.


Check out the [web interface](http://moderation.research.sfu.ca/) for this project.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── data_splitting.py
    │   │   └── preprocessor.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── baseline_models.py    
    │   │   ├── deeplearning_models.py
    │   │   └── feature_based_models.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


-----
### How to run the code

- Clone the repo and `cd` to your repo dir. 
- Set up environment. 
    - Edit `.env` and add your home path.  
    - Set up environment using: `source .env`
- Copy the data directory from the Data folder on the drive and put it your repo. 
- Run the models under `src/models`. 

