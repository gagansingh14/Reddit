# Reddit

It is often dicult to predict what factors directly contribute to the popularity of comments, which is also
a huge point of interest for fields such as marketing. In this project I will introduce the model for predicting the
popularity of comments on Reddit. More specifically, I will be analyzing the performance of linear regression
models on the popularity of Reddit comments. In the hypothesis I use features such as the actual comments, the
controversiality rating, whether the comment is a root (or a reply), and the number of replies to that comments
to predict the output vector, namely; the popularity score. I will train our classifiers using supervised learning,
and compare the gradient descent and closed form linear regression models, as well as compare models with the
top 160 words used, the top 60 words and with no text features, as well. Lastly, I will be adding two extra
features, a counter for the number of words used and the number of negative words used in the comment, which
will improve the model. I found that the closed form model with top 60 text features was the most optimal
model. Furthermore, the model which computed an mse of 1.26, is a better predictor than a normally or uniformly
distributed randomly generated model.

