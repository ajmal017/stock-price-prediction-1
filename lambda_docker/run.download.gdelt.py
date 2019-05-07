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

import zipfile
import boto3
import botocore

def handler(event, context):

	# Get yesterday's date, default get yesterday's events
	yesterday = datetime.now() - timedelta(days=1)
	current_date_str = yesterday.strftime("%Y%m%d")

	# Get start and end dates
	if event != None and "start_date" in event and "end_date" in event:
		start_date_str = event["start_date"]
		if start_date_str == None:
			start_date_str = current_date_str
		
		end_date_str = event["end_date"]
		if end_date_str == None:
			end_date_str = current_date_str

		start_date = datetime.strptime(start_date_str, '%Y%m%d')
	else:
		# Default current date
		start_date = yesterday
		start_date_str = current_date_str
		end_date_str = current_date_str

	# file names
	BUCKET_NAME = 'cumuluspython'              # bucket name
	tmp_path = '/tmp'
	tmp_zip_file_name = tmp_path + "/tmp.zip"
	tmp_sum_file_name = tmp_path + "/tmp_sum.csv"

	print "From " + start_date_str + " to " + end_date_str

	while start_date_str <= end_date_str:
		gdelt_download_url = "http://data.gdeltproject.org/events/" + start_date_str + ".export.CSV.zip"
		SUM_KEY = 'gdelt/' + start_date_str + '.sum.CSV' # object key

		# Download GDELT file
		testfile = urllib.URLopener()
		testfile.retrieve(gdelt_download_url, tmp_zip_file_name)

		print "Downloaded: " + gdelt_download_url

		# Unzip zip
		zip_ref = zipfile.ZipFile(tmp_zip_file_name, 'r')
		zip_ref.extractall(tmp_path)
		zip_ref.close()

		print "Unzipped: " + tmp_path

		# Read CSV
		tmp_csv_file_name = tmp_path + '/' + start_date_str + '.export.CSV'
		field_names = [ "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
					"Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode",
					"Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
					"Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode", "Actor2EthnicCode",
					"Actor2Religion1Code:", "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
					"IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
					"GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
					"Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_Lat", 
					"Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
					"Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type",       
					"ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long",
					"ActionGeo_FeatureID", "DATEADDED", "SOURCEURL" ]
		eventData = pd.read_csv(tmp_csv_file_name, names=field_names, header=None, delimiter="\t", index_col=None, na_filter=False)

		print "Read csv: " + tmp_csv_file_name

		# Summary by actors
		eventSumData1 = eventData.groupby(['SQLDATE', 'Actor1Geo_CountryCode', 'Actor1Type1Code'])['AvgTone', 'NumMentions'].agg('sum')
		eventSumData1.reset_index(inplace=True)
		eventSumData1.columns = ['SQLDATE', 'CountryCode', 'TypeCode', 'AvgTone', 'NumMentions']
		eventSumData2 = eventData.groupby(['SQLDATE', 'Actor2Geo_CountryCode', 'Actor2Type1Code'])['AvgTone', 'NumMentions'].agg('sum')
		eventSumData2.reset_index(inplace=True)
		eventSumData2.columns = ['SQLDATE', 'CountryCode', 'TypeCode', 'AvgTone', 'NumMentions']
		del eventData

		eventSumData3 = pd.concat([eventSumData1, eventSumData2])
		del eventSumData1
		del eventSumData2

		eventSumData4 = eventSumData3.groupby(['SQLDATE', 'CountryCode', 'TypeCode'])['AvgTone', 'NumMentions'].agg('sum')
		eventSumData4.reset_index(inplace=True)

		print "Sum csv"

		# Save summary to CSV
		eventSumData4.to_csv(tmp_sum_file_name)

		print "Wrote to csv: " + tmp_sum_file_name

		# Upload to S3
		s3 = boto3.resource('s3')

		try:
			s3.meta.client.upload_file(tmp_sum_file_name, BUCKET_NAME, SUM_KEY)

		except botocore.exceptions.ClientError as e:
			raise

		print "Uploaded: " + SUM_KEY

		# Remove temporary files
		os.remove(tmp_zip_file_name)
		os.remove(tmp_csv_file_name)
		os.remove(tmp_sum_file_name)
		
		print start_date_str + " finished"

		start_date = start_date + timedelta(days=1)
		start_date_str = start_date.strftime("%Y%m%d")

	return {'yay': 'done'}

handler({'start_date': '20180620', 'end_date': '20180630'}, None)
#handler({}, None)
