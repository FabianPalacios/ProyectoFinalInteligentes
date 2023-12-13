from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
uri = "mongodb+srv://Fabian:21jejAlo@proyectofinal.gaj5fmt.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection

# Select the database and collection
database_name = "InteligentesDB"
collection_name = "Inteligentes"

db = client.get_database(database_name)
collection = db.drop_collection(collection_name)

# Query the collection to retrieve data
query = {}  # You can customize the query to filter the results
results = collection.find(query)

# Print the results
for result in results:
    print(result)

# Close the MongoDB connection
client.close()