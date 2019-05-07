# imports
import os
import ctypes

for d, _, files in os.walk('lib'):
	for f in files:
		if f.endswith('.a'):
			continue
		ctypes.cdll.LoadLibrary(os.path.join(d, f))

import numpy as np
import pandas as pd

#we first need to make some extra imports
import json
from time import sleep
from datetime import datetime, date
from datetime import timedelta
import urllib

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

import zipfile
import boto3
import botocore

# S3 name and keys
SUM_BUCKET_NAME = 'cumuluspython'              # bucket name
STOCK_KEY = 'stock/stock.csv.zip'         # object key

# file names
tmp_path = '/tmp'
tmp_csv_file_name = tmp_path + "/tmp.csv"
tmp_sum_file_name = tmp_path + "/tmp_sum.csv"
tmp_stock_file_name = tmp_path + "/tmp_stock.csv"
tmp_sum_zip_file_name = tmp_path + "/tmp_sum.csv.zip"
tmp_stock_zip_file_name = tmp_path + "/tmp_stock.csv.zip"
tmp_model_zip_file_name = tmp_path + "/tmp_model.csv.zip"

predict_columns = ['EndPrice', 'TradedVolume']

# make 30 data on one line, plus next data
def make_30_data(stockData):
	length = len(stockData)
	columns_data = stockData.columns

	# create new data index and training data columns
	new_columns = ["idx"] + predict_columns
	for i in range(30):
		for column_name in columns_data:
			new_columns.append(column_name + "_" + "{0:02d}".format(i))

	# create new data frame
	all_dict = {}
	idx = 0

	# add new data into new data frame, start from line 30
	for i in range(30, length):
		all_dict[idx] = {}

		# X data
		for j in range(30):
			for column_name in columns_data:
				all_dict[idx][column_name + "_" + str(j)] = stockData.iloc[i-j-1][column_name]  # get i-1 to i-30 data

		# Y data
		for predict_column in predict_columns:
			all_dict[idx][predict_column] = stockData.iloc[i][predict_column]

		#all_dict[idx]["idx"] = stockData.index[i]
		idx += 1

	newStockData = pd.DataFrame.from_dict(all_dict).T
	#newStockData.set_index("idx",drop=True,inplace=True)
	#newStockData.sort_index(ascending=True,inplace=True)
	#print "30 historical data created:"
	#print len(newStockData)

	return newStockData

# make training and test data set
def split_data(newStockData):

	n, p = newStockData.shape
		
	# Create training and test data
	train_start = 0
	train_end = int(np.floor(0.8*n))
	test_start = train_end
	test_end = n
	data_train = newStockData.iloc[train_start:train_end]
	data_test = newStockData.iloc[test_start:test_end]
	#print str(len(data_train))
	#print str(len(data_test))

	return data_train, data_test

# training
def training(data_train, data_test, predict_columns):

	X_train = data_train[[c for c in data_train.columns if c not in predict_columns]]
	X_test = data_test[[c for c in data_test.columns if c not in predict_columns]]
	#print X_train.tail(5)
	#print X_test.tail(5)

	y_train = {}
	y_test = {}
	for predict_column in predict_columns:
		y_train[predict_column] = data_train[predict_column]
		y_test[predict_column] = data_test[predict_column]

	print "Begin to train. " + str(len(data_train))
	#print X_train.shape[0]
	#print X_test.shape[0]
	#print y_train["EndPrice"]
	#print y_test["EndPrice"]

	nn = {}

	for predict_column in predict_columns:

		nn[predict_column] = MLPRegressor(
			hidden_layer_sizes=(200,),  activation='relu', solver='lbfgs', alpha=0.001, batch_size='auto',
			learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=False,
			random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
			early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

		pipeline = Pipeline([
			('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
			('neural network', nn[predict_column])])

		nn[predict_column] = pipeline.fit(X_train, y_train[predict_column])

		# test
		print nn[predict_column].score(X_test, y_test[predict_column])
		y_test_predictions = nn[predict_column].predict(X_test)

		# export result for debug
		#yDict = {'y': y_test[predict_column], 'y_pred': y_test_predictions}
		#yDf = pd.DataFrame(yDict)
		#yDf.to_csv(predict_column + "_predict.csv")

	return nn

# Get start and end date range
def getDateRange(event):
	# Get yesterday's date, default get yesterday's events
	yesterday = datetime.now() - timedelta(days=1)
	current_date_str = yesterday.strftime("%Y-%m-%d")

	# Get start and end dates
	if event != None and "start_date" in event and "end_date" in event:
		start_date_str = event["start_date"]
		if start_date_str == None:
			start_date_str = current_date_str
		
		end_date_str = event["end_date"]
		if end_date_str == None:
			end_date_str = current_date_str

		start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
	else:
		# Default current date
		start_date = yesterday
		start_date_str = current_date_str
		end_date_str = current_date_str

	return start_date_str, end_date_str

# Check if the cvs files exist
def checkStockFiles():
	s3 = boto3.resource('s3')
	mybucket = s3.Bucket(SUM_BUCKET_NAME)
	response = mybucket.objects.filter(Prefix=STOCK_KEY)
	for obj in response:
		return True
	return False

# Fetch data from XETRA
def fetchData(start_date_str, end_date_str):

	start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

	df = pd.DataFrame()

	BUCKET_NAME = 'deutsche-boerse-xetra-pds'              # bucket name
	s3 = boto3.resource('s3')

	print "From " + start_date_str + " to " + end_date_str

	while start_date_str <= end_date_str:
		for i in range(24):
			hour_str = '{0:02d}'.format(i)
			KEY = start_date_str + '/' + start_date_str + '_BINS_XETR' + hour_str +'.csv' # object key

			try:
				# Download an hour data file
				s3.Bucket(BUCKET_NAME).download_file(KEY, tmp_csv_file_name)
			except botocore.exceptions.ClientError as e:
				if e.response['Error']['Code'] == "404":
					print("The object " + KEY + " does not exist.")
				else:
					raise

			# Read into data frame
			tmpDf = pd.read_csv(tmp_csv_file_name, header=0, index_col=None, na_filter=False)
			# Merge
			df = pd.concat([df, tmpDf])

			# Remove tmp files
			del tmpDf
			os.remove(tmp_csv_file_name)
		
		print start_date_str + " finished"

		start_date = start_date + timedelta(days=1)
		start_date_str = start_date.strftime("%Y%m%d")
	
	# Remove idx field
	#df.drop(["idx"], axis=1, inplace=True)

	# Save summary to CSV
	df.to_csv(tmp_sum_file_name)

	print "Wrote to csv: " + tmp_sum_file_name

	stockDf = df.groupby(['ISIN', 'Mnemonic', 'SecurityID'])['EndPrice', 'TradedVolume'].agg('mean')
	stockDf.reset_index(inplace=True)
	stockDf.to_csv(tmp_stock_file_name)

	print "Wrote to stock: " + tmp_stock_file_name

	# Zip
	with zipfile.ZipFile(tmp_sum_zip_file_name, "w") as newzip:
		newzip.write(tmp_sum_file_name, tmp_sum_file_name[len(tmp_path) + 1:])
		newzip.write(tmp_stock_file_name, tmp_stock_file_name[len(tmp_path) + 1:])

	# Upload to S3
	try:
		s3.meta.client.upload_file(tmp_sum_zip_file_name, SUM_BUCKET_NAME, STOCK_KEY)
	except botocore.exceptions.ClientError as e:
		raise

	print "Uploaded to S3."

	# Remove temporary files
	os.remove(tmp_sum_file_name)
	os.remove(tmp_stock_file_name)
	os.remove(tmp_sum_zip_file_name)

	return df, stockDf

# Download files from intermediate files
def downloadIntermediateData():

	s3 = boto3.resource('s3')

	try:
		s3.Bucket(SUM_BUCKET_NAME).download_file(STOCK_KEY, tmp_sum_zip_file_name)
	except botocore.exceptions.ClientError as e:
		if e.response['Error']['Code'] == "404":
			print("The object does not exist.")
		else:
			raise

	# Unzip
	zip_ref = zipfile.ZipFile(tmp_sum_zip_file_name, 'r')
	zip_ref.extractall(tmp_path)
	zip_ref.close()

	# Read into data frame
	df = pd.read_csv(tmp_sum_file_name, header=0, index_col=0, na_filter=False)
	stockDf = pd.read_csv(tmp_stock_file_name, header=0, index_col=0, na_filter=False)

	# Remove temporary files
	os.remove(tmp_sum_zip_file_name)
	os.remove(tmp_sum_file_name)
	os.remove(tmp_stock_file_name)

	print "Downloaded intermediate files from S3."

	return df, stockDf

# Remove intermediate data files
def removeIntermediateData():

	s3 = boto3.resource('s3')

	try:
		s3.Object(SUM_BUCKET_NAME, STOCK_KEY).delete()
	except botocore.exceptions.ClientError as e:
		if e.response['Error']['Code'] == "404":
			print("The object does not exist.")
		else:
			raise

# Main entry
def handler(event, context):
	# Get date range
	start_date_str, end_date_str = getDateRange(event)

	# Read intermediate data files or download them directly
	if checkStockFiles() == False:
		df, stockDf = fetchData(start_date_str, end_date_str)
	else:
		df, stockDf = downloadIntermediateData()

	# Remove useless columns
	df = df.drop(['Mnemonic'], 1)
	df = df.drop(['SecurityDesc'], 1)
	df = df.drop(['SecurityType'], 1)
	df = df.drop(['Currency'], 1)
	df = df.drop(['SecurityID'], 1)
	df = df.drop(['Date'], 1)
	df = df.drop(['Time'], 1)
	df = df.drop(['StartPrice'], 1)
	df = df.drop(['MaxPrice'], 1)
	df = df.drop(['MinPrice'], 1)
	df = df.drop(['NumberOfTrades'], 1)
	# ISIN, EndPrice, TradedVolume left

	# apply log to TradedVolume
	df['TradedVolume'] = df['TradedVolume'].astype('int64')
	df['TradedVolume'] = np.log10(df['TradedVolume'] + 1)

	# Upload models to S3
	s3 = boto3.resource('s3')

	# Iterate stock list
	idx = 0
	for index, row in stockDf.iterrows():
		stock_code = row["ISIN"]
		stockData = df.loc[df["ISIN"] == stock_code]

		if len(stockData) > 100:  # 100 means 70 records with 30 historic data
			# Remove useless columns
			stockData = stockData.drop(['ISIN'], 1)
			stockData.reset_index(drop=True, inplace=True)

			# EndPrice, TradedVolume left
			# print "stock data for " + stock_code
			# print stockData.tail(5)

			newStockData = make_30_data(stockData)

			# Split data set
			data_train, data_test = split_data(newStockData)

			print "Data ready for " + stock_code + "."
			#print data_train.tail(5)
			#print data_test.tail(5)

			# training
			nn = training(data_train, data_test, predict_columns)

			print "Training finished for " + stock_code + "."

			# save training result and upload to S3
			tmp_model_file_name = tmp_path + "/model" + stock_code + ".sav"
			joblib.dump(nn, tmp_model_file_name)

			MODEL_KEY = "stock/model" + stock_code + ".sav"         # object key

			try:
				s3.meta.client.upload_file(tmp_model_file_name, SUM_BUCKET_NAME, MODEL_KEY)
			except botocore.exceptions.ClientError as e:
				raise

			# Remove tmp file
			os.remove(tmp_model_file_name)

			# calculate for 5 stocks
			idx = idx + 1
			if idx > 5:
				break

	print "Uploaded to S3."

	# Remove intermediate data file
	removeIntermediateData()

	return {'yay': 'done'}

#handler(None, None)
handler({'start_date': '2018-08-21', 'end_date': '2018-08-30'}, None)
