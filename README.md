# ibm-recommend
Data Analysis and Articles Recommendations with IBM

### Table of contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

This should run by default with Anaconda installation with python 3.x. Requires Jupyter Lab/Notebook. 

Installing required libraries.

* Install external dependencies using `pip install -r requirements.txt`
* Install required local library by going in the root and running `pip install -e .`

After these are done, feel free to explore the notebooks. Ran first `01 - Data Wrangling.ipynb` to create the processed files.

### Recommendation Library

Doing `pip install -e .` in root will have `ibm_recommend` recommendation library available. Here's how to quickly use it.

```python
from ibm_recommend.recommendations import Recommender, CollaborativeRecommender, RankBasedRecommender, SVDRecommender

recommender = Recommender([
    SVDRecommender(),
    CollaborativeRecommender(),
    RankBasedRecommender()
])

recommender.fit(interactions)
```

After training the model you can do the following to get recommendations.

```python
# existing user
recommender.recommend(4484, rec_num=5)

# non-existing user
recommender.recommend(6000)
```

## Project Motivation<a name="motivation"></a>

This projects aim to analyze the current articles and user interactions with the datasets provided by IBM. There could be an opportunity here to further improve or increase the interactions of users by providing recommendations. We are going to walk you through the data and recommendations technique analysis.


## File Descriptions <a name="files"></a>

There 2 notebooks in this project. The `01 - Data Wrangling.ipynb` contains basic cleaning while `02 - Data and Recommendation Analysis` contains the analysis needed to answer the questions. Utility python and recommender codes are located at `ibm_recommend`.


## Results<a name="results"></a>

Main findings are located at `02 - Data and Recommendation Analysis.ipynb`.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

MIT License.