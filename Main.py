from surprise import Reader, Dataset, KNNBasic

# break data file down into an array full of strings
with open('./data.txt') as f:
    all_lines = f.readlines()
# load information from file into dataset using reader
reader = Reader(line_format='item user rating', sep=',', rating_scale=(1, 5))
data = Dataset.load_from_file('./data.txt', reader=reader)
# split dataset into n folds, can be changed
data.split(n_folds=5)
# using mean squared difference similarity measure here, with min_support set to 1 to consider only users who have at least 1 movie in common
sim_options = {'name': 'msd', 'user_based': False, 'min_support': 1}
trainingset = data.build_full_trainset()
# uses basic KNN algorithm to create a training set
algorithm = KNNBasic(sim_options=sim_options)
algorithm.train(trainingset)

# predict rating using item and user ID as input
userid = str(input("Please enter user ID: "))
itemid = str(input("Please enter movie ID: "))
print(algorithm.predict(userid, itemid))
