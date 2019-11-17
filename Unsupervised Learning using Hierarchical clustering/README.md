﻿<h1>Hierarchical clustering</h1>
<h3> Write a self-contained and fully functional Jupyter Notebook that: </h3>
<ol type = "a">
<li> Authenticates with the Twitter API </li>
<li> Using the Twitter Standard Search API, queries the most recent n = 50 tweets that have the #snl hashtag </li>
<li> Extracts the following 6 features from each tweet: 
<ol type = "i">
<li> followers_count of the user (number of followers of the user)</li>
<li> friends_count of the user (number of friends of the user)</li>
<li> statuses_count of the user (number of tweets of the user)</li>
<li> retweet_count of the tweet (number of retweets of the given tweet)</li>
<li> favorite_count of the tweet (number of favorites of the given tweet)</li>
<li> screen_name of the user (username of the given user) </li>
</ol>
</li>
<li> Using the first five features (i-v) previously built, performs hierarchical agglomerative clustering (with average linkage) on the data and plot the dendogram, using the screen name of the user (feature vi) as a label for the data.</li>
</ol>
<h4>libraries/packages:</h4>
<ul type="disc">
<li> Python Standard Library:<a href="url">https://docs.python.org/3/library/index.html</a></li>
<li> requests:<a href="url">https://requests.readthedocs.io</a></li>
<li> requests-oauthlib:<a href="url">https://requests-oauthlib.readthedocs.io</a></li>
<li> sklearn (aka scikit-learn):<a href="url">https://scikit-learn.org</a></li>
<li> scipy:<a href="url">https://www.scipy.org/about.html<a href="url"> </a></li>
<li> numpy </li>
<li> matplotlib</li>
<li> pandas</li>
</ul>
