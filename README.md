# N+1 Articles Difficulty Prediction
## Motivation

There is a problem with popular science news sites: you never know how exactly “popular” the article is, whether you have to have a PhD to understand it or if it is good for school children. And I thought it should be good to have an automatic system that decides it beforehand.



## Approach

There is a great popular science news site [N+1](https://nplus1.ru/) that assigns each article a difficulty rating based on human evaluation. So I thought I could scrape their website to make a dataset with article texts as an input and a numerical score as a target and train a machine learning model to predict the difficulty score based on the text itself.

