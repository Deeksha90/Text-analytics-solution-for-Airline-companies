import pandas as pd
# Import csv package to convert pandas dataframe to csv file
import csv
import pymongo
import warnings
warnings.filterwarnings('ignore')


# creating a connection with mongoDB client, to execute this the connection string needs to change as per respective mongoDB client.
client = pymongo.MongoClient(
    "mongodb+srv://dbuser:dbuser123@cluster0.6qv1c.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
)

# get the database
database = client['AirlineReview']


# get collection airlines_reviews
my_collection = database.get_collection("airline_reviews")
result = my_collection.find({})
Airlinedf = pd.DataFrame(list(result))

#deleted the id field which is not required for analysis
del Airlinedf['_id']
Airlinedf.head(5)

# exporting the dataframe to csv
Airlinedf.to_csv("Airline.csv", encoding='utf-8')

print(Airlinedf.head(5))