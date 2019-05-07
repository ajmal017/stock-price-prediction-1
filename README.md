A stock prediction program written in Python 2.7.

It is built in Docker and deployed in AWS Lamda.

This program has four parts (Lamda apps). 
* The first app downloads EUREX stock data daily from AWS S3 and updates and saves its model.
* The second downloads GDELT event data daily from AWS S3 and merges into the modal.
* The third accepts requests and does prediction and responses with the results.
* The last one tries to predict long-term price trend.  
