# fakenews_clf: A fake news classifier using Machine Learning techniques, deployed with Flask on Heroku

This project brings to life a fake news classifier, by deploying it using Flask on Heroku. It was also successfully deployed on Azure Portal with no adjustments.
The classification model is inspired by work from Aaron Edell, whose article inspired me for this project.


The [article title](https://towardsdatascience.com/i-trained-fake-news-detection-ai-with-95-accuracy-and-almost-went-crazy-d10589aa57c) is: <i> "I trained fake news detection AI with >95% accuracy, and almost went crazy" </i>

## The Web App
The WebApp can be found following [this link](https://fake-news-clf.herokuapp.com/).
It is deployed using Flask on Heroku, and can be deployed on Azure Portal with no further adjustments.
The Web App returns a probability a given article is fake news or not.

NB: The app might not be available at a given time, as Heroku turns them on and off to preserve resources.

### Layout
Layout is built using HTML & CSS.

1. A text box collects the user article: 
![alt text](https://github.com/mcsime92/fakenews_clf/blob/master/fakenewsclf-1.png "Text box for article")

2. Returned prediction: 
![alt text](https://github.com/mcsime92/fakenews_clf/blob/master/fakenewsclf-2.png "Returned prediction")

The user can then decide to test another article.
