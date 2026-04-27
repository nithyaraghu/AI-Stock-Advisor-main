# cosmos_client.py
from azure.cosmos import CosmosClient, exceptions
from config import settings

# Azure Cosmos DB settings
HOST = settings['host']
MASTER_KEY = settings['master_key']
DATABASE_ID = settings['database_id']
CONTAINER_ID = settings['container_id']

client = CosmosClient(HOST, MASTER_KEY)
database = client.create_database_if_not_exists(DATABASE_ID)
container = database.create_container_if_not_exists(
    id=CONTAINER_ID,
    partition_key="/id",  # Customize the partition key if needed
)

def create_user(data):
    container.create_item(body=data)
