# Code Summary #

`createDistros.py, createExponDistros.py`: These scripts generate distribution pairs to be used as the positive class score distribution and negative class score distribution for the calibrator comparison experiments. The distribution pairs are found to acheive a specific sample AUC.

`CalibratorMethodComparisonMono.py`: This code compares classifier calibration techniques using a single classifier score sampled from two distributions, one for the negative class and one for the positive class. The distributions are different combinations of generalized lambda distributions and normal distributions, with their location parameter set to achieve a specific sample AUCs.

`CalibratorMethodComparisonMulti.py`: This code compares classifier calibration techniques using two classifier scores sampled from four distributions. There are two distributions for each classifier, one for the negative class and one for the positive class. The distributions are different combinations of lambda distributions and normal distributions, with their location parameters set to achieve a specific sample AUCs.

`exponTest.py`: This code compares classifier calibration techniques using a single classifier score sampled from two exponential distributions, one for the negative class and one for the positive class, with their location parameter set to achieve a specific sample AUCs.

`recreateWeijie.py`: This code used the same calibration methods, distributions to simulate classifier scores, and calibrator parameters as Weijie in order to compare results to ensure consistency.

`isotpy/calibration.py`: The calibrator library, whose methods and classes are imported and used in the calibrator comparison experiments. Some are self implemented while others wrap sklearn classes.


`drebin/vectorizeDataSet.py`: Create a sparse numpy array from the raw drebin dataset

`terk/mktraintestsplit.py`: Make the train test split for classifiers in `terk/buildAndTestCLF.py`

`\[terk/drebin\]/buildAndTestCLF.py`: Train and test an SVM and Random Forest classifier on a real dataset, then save the classifier test scores to be used for the calibrator comparison experiment in `\[terk/drebin\]/RealDataSetTest.py`.

`\[terk/drebin\]/RealDataSetTest.py`: Using SVM and Random Forest classifiers trained on a real dataset, compare single-classifier and multi-classifier calibration techniques.

# This repo as a capsul on CodeOcean #
[https://codeocean.com/capsule/7882060/tree/v1](https://codeocean.com/capsule/7882060/tree/v1)

# References #
Yousef, W. A., Traore, I., & Briguglio, W. (2021), "Classifier calibration: with
implications to threat scores in cybersecurity", [arXiv Preprint,
arXiv:2102.05143](https://arxiv.org/abs/2102.05143).

# Citation #
Please, cite this work as:

```
@article{Yousef2021ClassifierCalibration-arxiv,
  author =       {Waleed A. Yousef and Issa Traore and William Briguglio},
  title =        {Classifier Calibration: with implications to threat scores in cybersecurity},
  year =         2021,
  journal ={arXiv Preprint, arXiv:2102.05143},
  url = {https://github.com/isotlaboratory/ClassifierCalibration-Code}
  primaryclass = {cs.LG}
}
```
