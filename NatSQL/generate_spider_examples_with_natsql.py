import json

spider_dev = json.load(open("data/dev.json","r"))
spider_train = json.load(open("data/train_spider.json","r"))

natsql_dev = json.load(open("NatSQLv1_6/dev-natsql.json","r"))
natsql_train = json.load(open("NatSQLv1_6/train_spider-natsql.json","r"))


assert len(spider_dev) == len(natsql_dev)
assert len(spider_train) == len(natsql_train)

for se, ne in zip(spider_dev,natsql_dev):
    se["NatSQL"] = ne["NatSQL"]
    se["sql"] = ne["sql"]

for se, ne in zip(spider_train,natsql_train):
    se["NatSQL"] = ne["NatSQL"]
    se["sql"] = ne["sql"]


json.dump(spider_dev,open("NatSQLv1_6/dev.json",'w'), indent=2)
json.dump(spider_train,open("NatSQLv1_6/train_spider.json",'w'), indent=2)

