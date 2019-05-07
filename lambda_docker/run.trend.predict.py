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
SUM_BUCKET_NAME = 'pwpython'              # bucket name
STOCK_KEY = 'stock/stock.xeur.csv'        # object key
GDELT_KEY = 'gdelt'

# file names
tmp_path = '/tmp'
tmp_csv_file_name = tmp_path + "/tmp.csv"
tmp_sum_file_name = tmp_path + "/tmp_sum.csv"
tmp_stock_file_name = tmp_path + "/tmp_stock.csv"
tmp_sum_zip_file_name = tmp_path + "/tmp_sum.csv.zip"
tmp_stock_zip_file_name = tmp_path + "/tmp_stock.csv.zip"
tmp_model_zip_file_name = tmp_path + "/tmp_model.csv.zip"
tmp_gdelt_file_name = tmp_path + "/tmp_gdelt.csv"
tmp_prediction_file_name = tmp_path + "/tmp_prediction.csv"

predict_columns = ['EndPrice', 'TradedVolume', 'AvgTone']

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
	print X_train.tail(5)
	print X_test.tail(5)

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
		#y_test_predictions = nn[predict_column].predict(X_test)

		# export result for debug
		#yDict = {'y': y_test[predict_column], 'y_pred': y_test_predictions}
		#yDf = pd.DataFrame(yDict)
		#yDf.to_csv(predict_column + "_predict.csv")

	return nn

# Predict for x days
def predict(data, predict_columns, days):

	X_data = data[[c for c in data_train.columns if c not in predict_columns]]
	X_data = data[len(X_data)]
	y_data = {}
	print X_data

	df = pd.DataFrame(columns = predict_columns)

	for i in range(days):
		for predict_column in predict_columns:
			y_data[predict_column] = nn[predict_column].predict(X_data)
		
		df.append(y_data)
		print y_data

		# Prepare for next day
		# EndPrice, NumberOfContracts, AvgTone
		new_columns.append(column_name + "_" + "{0:02d}".format(i))
		for j in range(29):
			for predict_column in predict_columns:
				X_data[predict_column + "_" + "{0:02d}".format(j)] = X_data[predict_column + "_" + "{0:02d}".format(j+1)]

		for predict_column in predict_columns:
			X_data[predict_column + "_30"] = y_data[predict_column]

	return df

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

# Fetch data from EUREX
def fetchData(start_date_str, end_date_str):

	start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

	df = pd.DataFrame()

	BUCKET_NAME = 'deutsche-boerse-eurex-pds'              # bucket name
	s3 = boto3.resource('s3')

	print "Fetch EUREX from " + start_date_str + " to " + end_date_str

	while start_date_str <= end_date_str:

		dayDf = pd.DataFrame()

		for i in range(24):
			hour_str = '{0:02d}'.format(i)
			KEY = start_date_str + '/' + start_date_str + '_BINS_XEUR' + hour_str +'.csv' # object key

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
			dayDf = pd.concat([dayDf, tmpDf])

			# Remove tmp files
			del tmpDf
			os.remove(tmp_csv_file_name)

		if len(dayDf) == 0:
			print start_date_str + " finished"
			start_date = start_date + timedelta(days=1)
			start_date_str = start_date.strftime("%Y-%m-%d")
			continue

		# Handle daily data
		dayDf.reset_index(inplace=True)		
		dayDf["idx"] = dayDf.index.values
		#print dayDf.tail(5)

		# Get contract summary of this day
		contractSumData = dayDf.groupby(['ISIN'])['NumberOfContracts'].agg('sum').to_frame()
		contractSumData.reset_index(inplace=True)
		contractSumData.columns = ['ISIN', 'NumberOfContracts']
		contractSumData.set_index(["ISIN"], inplace=True)
		#print contractSumData
		
		# Get end price index of this day
		lastContractData = dayDf.groupby(['ISIN'])['idx'].agg('max')
		lastContractDf = lastContractData.to_frame()
		lastContractDf.reset_index(inplace=True)
		lastContractDf.columns = ['ISIN', 'idx']
		#print lastContractDf.tail(5)

		# Get last end price from index
		for i in lastContractDf.index.values:
			idx = lastContractDf.at[i, "idx"]
			endPrice = dayDf.at[idx, "EndPrice"]
			#print str(endPrice)
			lastContractDf.at[i, "EndPrice"] = endPrice

		# Merge
		lastContractDf.set_index("ISIN", inplace=True)
		lastContractDf = lastContractDf.drop(columns=["idx"])
		#print lastContractDf.tail(5)
		sumData = pd.concat([contractSumData, lastContractDf], axis=1, sort=False)
		sumData["SQLDATE"] = start_date

		del dayDf
		del contractSumData
		del lastContractData
		del lastContractDf

		# SQLDATE, ISIN, EndPrice, NumberOfContracts
		df = pd.concat([df, sumData], sort=False)

		del sumData

		print start_date_str + " finished"
		start_date = start_date + timedelta(days=1)
		start_date_str = start_date.strftime("%Y-%m-%d")
	
	# Save summary to CSV
	df.to_csv(tmp_sum_file_name)

	# Upload to S3
	try:
		s3.meta.client.upload_file(tmp_sum_file_name, SUM_BUCKET_NAME, STOCK_KEY)
	except botocore.exceptions.ClientError as e:
		raise

	print "Uploaded to S3."

	# Remove temporary files
	os.remove(tmp_sum_file_name)

	return df

# Download files from intermediate files
def downloadIntermediateData():

	s3 = boto3.resource('s3')

	try:
		s3.Bucket(SUM_BUCKET_NAME).download_file(STOCK_KEY, tmp_sum_file_name)
	except botocore.exceptions.ClientError as e:
		if e.response['Error']['Code'] == "404":
			print("The object does not exist.")
		else:
			raise

	# Read into data frame
	df = pd.read_csv(tmp_sum_file_name, header=0, index_col=0, na_filter=False)

	# Remove temporary files
	os.remove(tmp_sum_file_name)

	print "Downloaded intermediate files from S3."

	return df

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

# Download GDELT data
def downloadGDELTData(start_date_str, end_date_str):
	df = pd.DataFrame()

	s3 = boto3.resource('s3')

	print "Download GDELT From " + start_date_str + " to " + end_date_str

	new_start_date = start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
	new_start_date_str = start_date.strftime("%Y%m%d")
	end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
	new_end_date_str = end_date.strftime("%Y%m%d")

	while new_start_date_str <= new_end_date_str:

		# Download
		try:
			s3.Bucket(SUM_BUCKET_NAME).download_file(GDELT_KEY + "/" + new_start_date_str + ".sum.CSV", tmp_gdelt_file_name)
		except botocore.exceptions.ClientError as e:
			if e.response['Error']['Code'] == "404":
				print "GDELT data not found: " + new_start_date_str
				new_start_date = new_start_date + timedelta(days=1)
				new_start_date_str = new_start_date.strftime("%Y%m%d")
				continue
			else:
				raise

		#print "Downloaded: " + new_start_date_str

		# Read into data frame
		tmpDf = pd.read_csv(tmp_gdelt_file_name, header=0, index_col=None, na_filter=False)
		#print tmpDf.tail(5)
		# Calculate summary
		sumData = tmpDf.groupby(['SQLDATE'])['AvgTone'].agg('sum').to_frame()
		sumData.reset_index(inplace=True)
		sumData.columns = ['SQLDATE', 'AvgTone']
		# Merge
		df = pd.concat([df, sumData])

		# Remove tmp files
		del sumData
		os.remove(tmp_gdelt_file_name)

		print new_start_date_str + " finished"

		new_start_date = new_start_date + timedelta(days=1)
		new_start_date_str = new_start_date.strftime("%Y%m%d")

	# Calculate summary
	#print df.tail(5)
	sumDf = df.groupby(['SQLDATE'])['AvgTone'].agg('sum').to_frame()
	sumDf.reset_index(inplace=True)
	sumDf.columns = ['SQLDATE', 'AvgTone']
	sumDf["SQLDATE"] = pd.to_datetime(sumDf["SQLDATE"], format="%Y%m%d")
	return sumDf

# Main entry
def handler(event, context):
	# Get date range
	start_date_str, end_date_str = getDateRange(event)

	# Get GDETLT data
	gdeltDf = downloadGDELTData(start_date_str, end_date_str)

	# Read intermediate data files or download them directly
	if checkStockFiles() == False:
		df = fetchData(start_date_str, end_date_str)
	else:
		df = downloadIntermediateData()

	df.reset_index(inplace=True)
	df.columns = ['ISIN', 'NumberOfContracts', 'EndPrice', 'SQLDATE']
	df["SQLDATE"] = pd.to_datetime(df["SQLDATE"], format="%Y-%m-%d")

	# print df.tail(5)

	# Get all stock list
	stockDf = df.groupby(['ISIN'])['NumberOfContracts'].agg('sum').to_frame()
	stockDf.reset_index(inplace=True)
	stockDf.columns = ['ISIN', 'NumberOfContracts']

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

			# SQLDATE, EndPrice, NumberOfContracts left
			# print "stock data for " + stock_code
			# print stockData.tail(5)
			#print stockData.dtypes

			# Add GDELT tone
			for indexStock, row in stockData.iterrows():
				theDate = row["SQLDATE"]
				avgTone = gdeltDf[gdeltDf['SQLDATE'] == theDate]['AvgTone']
				stockData.at[indexStock, 'AvgTone'] = avgTone.values[0]
			print stockData.tail(5)

			# Remove useless columns
			stockData = stockData.drop(['SQLDATE'], 1)

			newStockData = make_30_data(stockData)

			# Split data set
			data_train, data_test = split_data(newStockData)

			print "Data ready for " + stock_code + "."
			#print data_train.tail(5)
			#print data_test.tail(5)

			# training
			nn = training(data_train, data_test, predict_columns)

			print "Training finished for " + stock_code + "."

			predictedDf = predict(newStockData, predict_columns, 5)
			predictedDf.to_csv()

			# Upload to S3
			MODEL_KEY = "stock/predict." + stock_code + ".csv"         # object key

			try:
				s3.meta.client.upload_file(tmp_prediction_file_name, SUM_BUCKET_NAME, MODEL_KEY)
			except botocore.exceptions.ClientError as e:
				raise

			# Remove tmp file
			os.remove(tmp_prediction_file_name)

			# calculate for 5 stocks
			idx = idx + 1
			if idx > 5:
				break

	# Remove intermediate data file
	removeIntermediateData()

	return {'yay': 'done'}

#handler(None, None)
handler({'start_date': '2018-06-20', 'end_date': '2018-08-30'}, None)
