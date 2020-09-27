#import numpy as np
#import pandas as pd
#
## Importing the dataset
#dataset = pd.read_csv('BCLL.txt',sep='\t')
#X = dataset.iloc[:,[0,1,2,3,4,5,6,7,8,9,11]].values
#y = dataset.iloc[:,21].values
#print(X.shape)
#print(y.shape)


import numpy as np
import matplotlib.pyplot as plot
from matplotlib.pyplot import style
style.use("seaborn-darkgrid")
#from matplotlib import style
#style.use("ggplot")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score


#data got from user
# all_data = np.array([[1, 2], [5,8], [1.5, 1.8] ,[8,8], [9,11], [10,1], [7,5], [9,2], [3,7]])
# a = all_data
#
# #data access from unlabelled data
read_file=open("BCLL.txt",'r')
#next(read_file)
# # data access from labeled data
# #read_file=open("iris_org.txt",'r')
#
#
#from itertools import islice
#with open("BCLL.txt",'r') as read_file:
#    for line in islice(read_file,1,None):
#        print(read_file)

#new_data=read_file.readlines()[1:]
read_content= read_file.read()
all_data= read_content.splitlines()
#print (all_data)
No_data_points=len(all_data)
print ("Number of the data points ", No_data_points)
features = all_data[0].split("\t")
print ("features of the data points ", features)
#

#For unlabelled data
No_of_dimension = len(features)
##
## #For Labelled data
## No_of_dimension = len(features)-1
##
##
##
print ("Number the dimensions", No_of_dimension)
#for lines in all_data[1:]:
#    array=lines.split('\t')[2:No_of_dimension]
#print (array)
#features = array[0].split("\t")
#print ("features of the data points ", features)
##
#
##For unlabelled data
#No_of_dimension = len(features)
###
### #For Labelled data
### No_of_dimension = len(features)-1
###
###
###
#print ("Number the dimensions", No_of_dimension)
##
a=np.zeros((No_data_points,No_of_dimension-2))
## true_label = []
counter = 0

for lines in all_data[1: ]:
    values=lines.split('\t')[2:No_of_dimension]
#    print (values)
    for i in range(0,(No_of_dimension-2)):
        a[counter][i]= values[i]
    #true_label.append(int(values[No_of_dimension]))     #For labelled data
    counter+=1



import random, math     
def eucledianDistance(arg1,arg2):
    distance=0
    distanceRoot=0
    for i in range(len(arg1)):
        distance= distance + (arg1[i]-arg2[i])**2
    distanceRoot=(distance)**0.5
    return distanceRoot
def toleranceLevel(arg1,arg2):
    level=0
    for i in range(len(arg1)):
        level= level + abs(((arg2[i]-arg1[i])/arg1[i]))
    return level

variable=0
a1=[1,3,4,3]
a2=[1,5,3,3]
variable = eucledianDistance(a1,a2)
print ("distance is",variable)

from random import randrange
n=int((No_data_points)**0.5)
print("n no is",n)
#for i in range(1):
    
ncluster=randrange(2,6)
print("random no is",ncluster)
kmeans = KMeans(n_clusters=ncluster)
kmeans.fit(a)
print (kmeans)
np.random.seed(1)
counter=0
newArray=np.zeros((No_data_points,No_of_dimension-2))  
dict={}
#    for i in range(0,ncluster):
##        dict[i]=[i,np.zeros((No_data_points,No_of_dimension-2))]
##        filename="Cluster_ " %i% ".txt"
#        filename_cluster="Cluster_ "+ str(i)
#        f= open(filename_cluster + ".txt","a+")
for i in range(0,kmeans.max_iter) :
    centroids=np.zeros((ncluster,No_of_dimension-2))  
    centroidsNew=np.zeros((ncluster,No_of_dimension-2))
    for j in range(0,ncluster):
        centroids[j]=a[j]
        
#    print (centroids)
    distances=np.zeros((ncluster))
#    print (label)
    for k in range(0,No_data_points):
        for j in range(0,ncluster):
            distances[j]=eucledianDistance(a[k],centroids[j])
#        print("distances are",distances) 
#       minIndex = distances.choose(min(distances))
        minIndex = np.argmin(distances)
#        print("k index: ",k,"is assigned to cluster: centroid : ",minIndex)
        filename_cluster="Cluster_ "+ str(minIndex) + ".txt"
        f= open(filename_cluster,"a+")
        with open(filename_cluster,"a+") as f1:
            np.savetxt(f1,np.array(a[k]),delimiter='\n', newline='\t' )
            f1.write("\n")
#        print("array:",dict[i])
#        print("array:",a[k])
    for j in range(0,ncluster):      
        counter = 0
        filename_cluster="Cluster_ "+ str(j) + ".txt"
        f= open(filename_cluster,"r+")
        read_content= f.read()
        all_cluster_points= read_content.splitlines()
        No_cluster_points=len(all_cluster_points)
        cluster_datapoints=np.zeros((No_cluster_points,No_of_dimension-2))
        for lines in all_cluster_points:
            values=lines.split('\t')
            for i in range(0,(No_of_dimension-2)):
                cluster_datapoints[counter][i]= values[i]
            counter+=1
        centroidsNew[j]=np.mean(cluster_datapoints,axis=0,keepdims=1)
    for l in range(0,ncluster):
        x1 = centroids[l]
        x2 = centroidsNew[l]
        level=toleranceLevel(x1,x2)
        print("level",level)
        if  level>0.001 :
            convergence = False
        else:      
            convergence = True
    if convergence :
        print("break")
        break