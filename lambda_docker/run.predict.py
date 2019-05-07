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
from boto3.dynamodb.conditions import Key, Attr
from decimal import Decimal

# S3 name and keys
SUM_BUCKET_NAME = 'cumuluspython'              # bucket name

# DynamoDB region and table
DYNOMODB_REGION = 'ap-southeast-2'
DYNAMODB_TABLE = 'trade_activities'

# file names
tmp_path = '/tmp'
tmp_model_file_name = tmp_path + "/tmp_model.csv"

predict_columns = ['EndPrice', 'TradedVolume']

# Get the timestamp to predict
def getTimestamp(event):
	# Get start and end timestamp
	if event != None and "timeStamp" in event:
		timestamp_start = event["timeStamp"]

	if timestamp_start == None:
		timestamp_start = datetime.now()

	return timestamp_start

# Download files from model files
def downloadModels(stock_code):

	MODEL_KEY = "stock/model" + stock_code + ".sav"         # object key

	s3 = boto3.resource('s3')

	try:
		s3.Bucket(SUM_BUCKET_NAME).download_file(MODEL_KEY, tmp_model_file_name)
	except botocore.exceptions.ClientError as e:
		if e.response['Error']['Code'] == "404":
			print("The object does not exist.")
		else:
			raise

	print "Downloaded model files from S3."

	return

# Download 30 data from now backward
def downloadHistoricalData(stock_code, timestamp_start):

	predict_dict = {}
	item_count = 0

	print "Start to get data from DynamoDB: " + stock_code + " & " + str(timestamp_start)

	# Connect dynamodb and scan
	dynamodb = boto3.resource('dynamodb', region_name=DYNOMODB_REGION)
	table = dynamodb.Table(DYNAMODB_TABLE)

	fe = Key('product_isin').eq(stock_code) & Key('time_stamp').lte(timestamp_start)
	pe = "end_price, volume_of_trades, time_stamp"
	response = table.scan(
		FilterExpression=fe,
		ProjectionExpression=pe)

	# Save result into dict
	for item in response['Items']:
		item_count = item_count + 1
		predict_dict["EndPrice_" + str(item_count)] = item['end_price']
		# apply log to TradedVolume
		predict_dict["TradedVolume_" + str(item_count)] = np.log10(item['volume_of_trades']) + 1

		#print "end_price: " + str(item['end_price'])
		if item_count >= 30:
			break

	# Save more pages until 30 data got
	while 'LastEvaluatedKey' in response and item_count < 30:
		response = table.scan(
			FilterExpression=fe,
			ProjectionExpression=pe,
			ExclusiveStartKey=response['LastEvaluatedKey']
			)

		for item in response['Items']:
			item_count = item_count + 1
			predict_dict["EndPrice_" + str(item_count)] = item['end_price']
			predict_dict["TradedVolume_" + str(item_count)] = item['volume_of_trades']
			if item_count >= 30:
				break

	if item_count < 30:	
		print "Historical data too less: " + str(item_count)
		return None

	predict_dict["idx"] = 30

	predictData = pd.DataFrame(predict_dict, index=[0])
	predictData.set_index("idx",drop=True,inplace=True)
	predictData.sort_index(ascending=True,inplace=True)

	print "30 prediction data created."
	#predictData.to_csv("/tmp/tmp.csv")
	#print predictData.tail(5)

	return predictData

def uploadResult(stock_code, timestamp_start, price_predicted):

	# Save predicted prices to DB
	timestamp_new = timestamp_start + 60
	dynamodb = boto3.resource('dynamodb', region_name=DYNOMODB_REGION)
	table = dynamodb.Table(DYNAMODB_TABLE)
	response = table.put_item(
		Item={'predicted_end_price': Decimal(str(price_predicted)),
				'time_stamp': timestamp_new,
				'product_isin': stock_code
			}
		)

# Main entry
def handler(event, context):
	# Get input parameters
	stock_code = event["ISIN"]
	timestamp_start = getTimestamp(event)

	# Download model data files
	downloadModels(stock_code)

	# load the model from disk
	nn = joblib.load(tmp_model_file_name)

	# Download historic from DB
	predictData = downloadHistoricalData(stock_code, timestamp_start)
	if predictData is None:
		return {'error': 'No or too less data'}

	# Predict price
	price_predicted = nn["EndPrice"].predict(predictData)
	print predictData
	print(price_predicted[0])

	uploadResult(stock_code, timestamp_start, round(price_predicted[0], 2))
	print "Uploaded to DB."

	return {'yay': 'done'}

#handler(None, None)
handler({'ISIN': 'DE0005439004', 'timeStamp': 1496296860001}, None)
