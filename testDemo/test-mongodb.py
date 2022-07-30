from pymongo import MongoClient


def query_count(collection):
    query_pics = collection.find()
    count = 0
    for i in query_pics:
        count += 1
    return ("{:s} colletion has {:d} documents".format(str(collection.name), count))

def drop_col(collection):
    print("drop all documents in {:s}".format(str(collection.name)))
    collection.drop()

def refresh(collection):
    query_count(collection)
    drop_col(collection)
    print("after drop.", query_count(collection))


print("Connect to mongoDB")
mongodbUri = 'mongodb://user3:myadmin@101.35.252.209:27017/admin'
# user1，myadmin
# user2，myadmin
# user3，myadmin
client = MongoClient(mongodbUri)
mydb = client['images']
pictures = mydb["pictures"]
respo = mydb["responses"]
print("Successfully connect to mongoDB\n==========\n")

# refresh(pictures)

print(query_count(pictures))
# drop_col(pictures)
# query_count(pictures)
#
print(query_count(respo))
# drop_col(respo)
# query_count(respo)
