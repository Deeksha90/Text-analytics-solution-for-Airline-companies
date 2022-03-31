# Text-analytics-solution-for-Airline-companies

In this project, we have a dataset of real customer review collected from Skytrax (www.airlinequality.com), a major website for customers to evaluate various airline companies. This dataset consists of 41,396 review entries and each of the entry has rating of a particular airline company by an author. We have stored this data in MongoDB database. We need to use PyMongo to access the data from MongoDB. The next step is to perform sentiment analysis on the data using a machine learning based classification (using nltk). Next, we need to construct a rating network between author and the airlines company they rated. Py2Neo is used to construct the network in Neo4j. Final step is to do a deeper analysis on the airlines and compare each airline with its competitors

# Data Source

The information we have is based on actual customer reviews obtained from Skytrax. This dataset contains 41,396 review entries, with each entry containing an author's rating of an airline company. An author is both a rater and a customer in this context. The information is saved in a MongoDB database. The overall rating is based on a scale of 1 to 10, with a higher level indicating greater satisfaction. Individual aspects (such as cabin crew, seat comfort, food and beverage, and so on) are rated on a scale of 1 to 5. The "recommended" property indicates whether or not the author would recommend this airline to others. The data is in the json format.

# MongoDB Database storage and PyMongo retrival:

• Created Airline Review database and imported reviews json into airline reviews json.
• Installed all the requirements example PyMongo package.
• Using connection string, mongo dB connection is done.
• This string is specific to the mongo dB connection it is making.
• Once the database and collection connections have been established,
• Find query is used to retrieve all the data in the collection, which is then converted to a pandas data frame.
• Removing the id field from the data frame because it isn't needed in the analysis and storing it as a csv file for further analysis.

# Sentimental Analysis: 
Here we pose sentiment analysis as binary classification problem with two classes – Positive and Negative. 
We check for the labels in Recommended column; If Recommended attribute is 1 we tag it as positive otherwise negative. We apply Naïve Bayes Classification Model for this Analysis and prediction.

# Network Based Analysis in Neo4j
Constructed a rating network between authors and the airline companies that they rated. Such rating network captured who rated which airline companies. Created this netowrk in Neo4j. 

# Report and Dashboard creation in Tableau:
We created reports and dashboard using Tabpy API to caluculate sentiments from ReviewContent column.Also created dashboard for the same.


# Conclusion
After completing the sentimental analysis we were able to obtain 85% F-score.
