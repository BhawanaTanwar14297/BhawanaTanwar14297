#description: Program to detect clusters for a variable. Input file: Json with columns "Date" and "DataValue" in record form. Output: Json with columns "Date", "Centroid", "Opacity" and "Frequency" in record form.
#author: Bhawana Tanwar
#python Version : 2.7.14

# import the libraries
import sys
import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
import json
import ast
import datetime
from numpy import mean
from numpy import std

class CLSGetBubbleChartData:
    #functions
    #function to find the date. Input Parameters: DataFrame, Name of Column and centroid

    def find_closest(self,df, var,centroid):
    #calculating distance
        dist = (df[var] - centroid[0]).abs()
        df['dist'] = dist
        d = pd.DataFrame()
        #detecting the date for minimum distance
        d['date'] = df[df.dist == df.dist.min()]['Date']
        d.reset_index(drop = True, inplace = True)
        return ((d['date'][len(d)//2]))
   

        #delete Outliers
    def deleteOutlier(self, data):
    #    Outlier is detected keeping 3 Standard Deviations from the Mean:i.e. 99.7% of data
        data_mean, data_std = mean(data['DataValue']), std(data['DataValue'])
        data.reset_index(drop = True,inplace = True)
        # identify outliers
        cut_off = data_std * 3
        lower, upper = data_mean - cut_off, data_mean + cut_off
        outliers = [ i for i in range(0,len(data['DataValue'])) if float(data['DataValue'][i]) < lower or float(data['DataValue'][i]) > upper]
        if outliers:
            data.drop([outliers[0]], axis=0, inplace=True)
            return([data,outliers])
        else:
            return([data,''])



    def optimumK(self,X,NumObs):
        #calculating within sum of squares for clusters
        wcss = []
        
        if NumObs<=30:
            maxCluster = NumObs//2
        elif NumObs>=30:
            #for more than 30 observations max number of clusters is set as 10 as per discussion with Sudeep 
            maxCluster = 10
        if maxCluster>=10:
            n = 10
        else:
            n =maxCluster
      
        for i in range(1, 10):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        
        #calculating first Difference
        c = []
        for i in range(0,len(wcss)-1):
            if wcss[i]<sum(wcss)/len(wcss) and i!=0:
                c.append(i)  
        return(c[0])


    #calling
    def GetBubbleChartData(self,path,ID):
        #Reading data from json and manipulating it for further processing 
        filejson = path+'/'+ID+'.json'
        f = open(filejson)
        # f = open('./pathNew.json')
        data = json.load(f)
        data = ast.literal_eval(json.dumps(data)) 
        df1 = pd.DataFrame(data[data.keys()[0]],columns = ['Date','DataValue'])
        df = pd.DataFrame()
        res = self.deleteOutlier(df1)
        df = res[0]
        outliers = res[1]
        df.reset_index(drop = True, inplace=True)
        if df.size >10:
            #clustering and preparing output
            A = df
            x = A
            #name of the variable for clustering
            var = "DataValue"
            #convertin column to nparray for calculation
            a = np.array(A[var]).reshape(-1, 1)
            #calling function for optimum number of clusters
            clusOptNum = self.optimumK(a,df.size)
            kmeans = KMeans(n_clusters = clusOptNum, random_state=0)
            #fitting the model
            y_predicted = kmeans.fit_predict(a)
            df['Labels'] = y_predicted
            datlis = []
            for i in range(0, len(np.unique(np.array(y_predicted)))):
                d = pd.DataFrame()
                d[str(i)+'cluster'] = df[df.Labels == i]['Date']
                indDate = len(d[str(i)+'cluster'])//2
                datlis.append(d[str(i)+'cluster'][indDate])            
            #getting centroids 
            clusters = kmeans.cluster_centers_
            centroid = (kmeans.cluster_centers_)
            frequency = []
            frequency1 = []
            #extracting date,ferequency for centroids
            x['Date'] = df['Date']
            for i in range(0,4):
                c = 0
                for j in range (0,len(y_predicted)):
                    if y_predicted[j] == i:
                        frequency.append(centroid[i][0])
                        c = c+1
                frequency1.append(c)
            centroidList = []
            for i in centroid:
                centroidList.append(i[0])
            c = []
            for i in centroid:
                c.append(i[0])
            #creating dictionary for sumping in json
            outputlist = []
            #calculating opacity
            maxfreq = max(frequency1)
            op = []
            for i in frequency1:
                op.append(float(i)/float(maxfreq))
            #creating dictionary value
            for j in range(0,len(c)):
                if frequency1[j]!=1:
                    outputlist.append({'Date':datetime.datetime.strptime(str(datlis[j]), "%m/%d/%Y").strftime("%Y-%m-%d"),'Centroid':float(c[j]) , 'Opacity':float(op[j]), 'Frequency':int(frequency1[j])})
            #defining key and value for dictionary
            dictout = {"Table": outputlist}
            json_str = json.dumps(outputlist)
            #saving records to json 
            with open(path +'/'+ ID+'_answer'+'.json', 'w') as f:
                json.dump(dictout, f)
            return(ID+"_answer")    
        else:
            #less than 10 observations 
            return("")


#creating object of class
CLSGetBubbleChartDataObj = CLSGetBubbleChartData()
#capturing input from command line and casting to integer
path = 'E:\TSS\Application\WebServices_VS2013\TSSDigitizationDBWCFService\TempFiles'#str(sys.argv[1])
ID = '68c461ac-01ea-4396-bdad-b8c159c2d0dd'#str(sys.argv[2])
#y = int(sys.argv[2])
ANSJson = CLSGetBubbleChartDataObj.GetBubbleChartData(path,ID)
#printing result on console
print(ANSJson)
