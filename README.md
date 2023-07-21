# A-Fake-Real-News-Detection-for-Vaccination
This repository contains all the files and datasets that have contributed to the research, "A Fake/Real News Detection for Vaccination"

To reproduce the experiment, you would actually need to follow the prescribed labels of the labelled Python files from 1 to 5 according to the instructions.

The datasets collected and compiled for the experiment are located inside Cap2Datasets

Ensure that the current working directory is set to the same directory as the Python files to work properly.

## Metadata

After compilation and processing, the dataset contains 37660 records with 4 attributes.
| Attributes             | Description                                                                |
| ----------------- | ------------------------------------------------------------------ |
| title | Headlines of the articles/posts in the dataset. Not all datasets have this. |
| content | The main content of the dataset. Typically containing the large text of string |
| source | Origin of the news/information. Categorised to either “Articles” or “Social Media” |
| label | 1 denotes that the news is true and 0 denotes the news is fake. |


## Citations

### Dataset Included

```
@misc{patwa2020fighting, title={Fighting an Infodemic: COVID-19 Fake News Dataset}, author={Parth Patwa and Shivam Sharma and Srinivas PYKL and Vineeth Guptha and Gitanjali Kumari and Md Shad Akhtar and Asif Ekbal and Amitava Das and Tanmoy Chakraborty}, year={2020}, eprint={2011.03327}, archivePrefix={arXiv}, primaryClass={cs.CL} }
```
```
Koirala, Abhishek (2021), “COVID-19 Fake News Dataset”, Mendeley Data, V1, doi: 10.17632/zwfdmp5syg.1
```
```
@inproceedings{shahifakecovid,
title={Fake{C}ovid -- A Multilingual Cross-domain Fact Check News Dataset for COVID-19},
author={Shahi, Gautam Kishore and Nandini, Durgesh},
booktitle={Workshop Proceedings of the 14th International {AAAI} {C}onference on {W}eb and {S}ocial {M}edia},
year = {2020},
url = {http://workshop-proceedings.icwsm.org/pdf/2020_14.pdf}
}
```
```
@misc{cui2020coaid,
    title={CoAID: COVID-19 Healthcare Misinformation Dataset},
    author={Limeng Cui and Dongwon Lee},
    year={2020},
    eprint={2006.00885},
    archivePrefix={arXiv},
    primaryClass={cs.SI}
}
```
```
Sumit Banik. (2020). COVID Fake News Dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4282522
```
```
Li, S. (2020, May 27). Explore COVID-19 Infodemic. Medium. https://towardsdatascience.com/explore-covid-19-infodemic-2d1ceaae2306
```
```
Shapiro, J., Oledan, J., & Siwakoti, S. (2020). ESOC COVID-19 Misinformation Dataset. Empirical Studies of Conflict. https://esoc.princeton.edu/publications/esoc-covid-19-misinformation-dataset
```
```
Kim, J., Aum, J., Lee, S., Jang, Y., Park, E., & Choi, D. (2021). FibVID: Comprehensive fake news diffusion dataset during the COVID19 period. Telematics and Informatics, 64, 101688. https://doi.org/10.1016/j.tele.2021.101688
```
```
Memon, Shahan Ali, & Carley, Kathleen M. (2020). CMU-MisCov19: A Novel Twitter Dataset for Characterizing COVID-19 Misinformation [Data set]. 5th International Workshop on Mining Actionable Insights from Social Networks (MAISoN) at CIKM 2020, Online. Zenodo. https://doi.org/10.5281/zenodo.4024154
```
```
Inuwa-Dutse, I., & Ioannis Korkontzelos. (2020). A curated collection of COVID-19 online datasets. https://doi.org/10.48550/arxiv.2007.09703
```
```
@inproceedings{zhou2020recovery,
  title={ReCOVery: A Multimodal Repository for COVID-19 News Credibility Research},
  author={Zhou, Xinyi and Mulay, Apurva and Ferrara, Emilio and Zafarani, Reza},
  booktitle={Proceedings of the 29th ACM International Conference on Information & Knowledge Management},
  pages={3205--3212},
  year={2020}
}
```
```
Shakshi Sharma, Ekanshi Agrawal, Rajesh Sharma, & Anwitaman Datta. (2022). FaCov Dataset: COVID-19 Viral News and Rumors Fact-Check Articles Dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5854656

```
### CT-BERT Model
```
Martin Müller, Marcel Salathé, and Per E Kummervold. 
COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter. 
arXiv preprint arXiv:2005.07503 (2020).
```
