#coded by shibinmak

from pyspark.sql import Session
from pyspark.ml.regression import LinearRegression
#import for data preprocessing
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler,StringIndexer,OneHotEncoder
#start session
spark =SparkSession.builder.appName('Linear Regressoin').getOrCreate()

#read data
data = spark.read.csv('file.csv',header=True,inferSchema=True)
#check data,schema,identify features
data.take(1)[0].asDict() #show details as dictionary
data.printschema
data.columns

"""data processing"""
#identify categorical data

#analyse categorical data
data.groupBy('Categorical_data').count().show()

#stringindexing(based on most occurance )

indexer =StringIndexer(inputCol='Categorical_data',outputCol='string_indexed')
#fit and transform categorical data
indexed = indexer.fit(data).transform(data)

#remove dummy varibale by onehotencoder
encoder = OneHotEncoder(dropLast=True,inputCol='string_indexed',outputCol='encoded')
encoded = encoder.transform(indexed)

#assembling features into a single vector (MLlib demands this format)
assembler = VectorAssembler(inputCols=['feature1','feature2','feature3','encoded'],outputCol='features')
featcols=['feature1','feature2','feature3','encoded'] #will be useful in future analysis
output = assembler.transform(encoded) #assembles feature column(dense vector or sparse)vector with data

#select Label and feature column
final = output.select(['crew','features'])

#split data to test and train data
train_data,test_data = final.randomSplit([0.7,0.3])

--------machine learning ------------
#initiate LinearRegressorObject
regressor = LinearRegression(featuresCol='features',labelCol='label',predictionCol='predicted label')
#fit train_data
model = regressor.fit(train_data)

#evaluate model by calling evaluate() method which will give number of option to evaluate model ,such as r2,rootMeanSquaredError
eval_model = model.evaluate(test_data)
eval_model.rootMeanSquaredError
eval_model.r2

#prediction of test_data by model 
prediction = model.transform(test_data.select('features'))

#check coefficients and intercepts of model (can decide which feature has more effect on label(independent variable)
coeff=model.coefficients  
intercept= model.intercept
a= zip(featcols,coeff)
b= set(a)
for x,y in b:
    print(x ,':' ,y)
    print('\n')
#analysis only
from pyspark.sql.functions import corr
data.select(corr('feature1','label')).show()
data.select(corr('feature1','feature2')).show()


