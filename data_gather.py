"""
Filename: data_gather.py
Author: Jayden Jardine
Date: 2025-05-07
Version: 1.0
Description: This script gets data from a data source @TODO and formats it such that its ready to pass in the Bert uncased tokenizer.
             This file is also meant to format the inputs and outputs to use with a pytorch dataset.
"""


from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
from datetime import date, datetime, timedelta
import pandas as pd
import csv



def tokenize_inputs(inputs, max_length = 128 ):

    '''
    Summary: Uses the Hugging face BertTokenizer inherited from [PreTrainedTokenizer] to tokenize an aray of strings

    Inputs: an array of strings

    outputs: {input_ids:[[]..[]], token_type_ids:[[]..[]] ,attention_mask:[[]..[]] } 
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized = tokenizer(inputs,truncation=True, padding=True, max_length=max_length) # max length of 128 is valid for use case
    return tokenized # returns {tokenized-inputs:[], attention-masks:[]}


def load_data():
    '''
    load_data():
    Input: None
    Output: (dict,dict) both of matching sizes 

    Summary:
        * read in data from CSV on news headlines, then organize as dict by ticker symbol 
        * - do - for price data
        * Ensure both datasets have the same tickers
    '''
    # open csv

    news_mapping = {}
    price_mapping = {}

    '''
    Following method opens up the csv holding the stock news headlines
        and organizes by ticker symbol
    format:
        news_mapping['<ticker>'] = {
            "index": <string> ~ easy cast to int if needed 
            "headline": <string>
            "url": <string>
            "publisher": <string>
            "date": <string>
            "stock": <string>
    }
    '''
    with open('archive/raw_analyst_ratings.csv', mode='r',newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            item = dict(row)
            if row['stock'] not in news_mapping:
                news_mapping[row['stock']] = []
            news_mapping[row['stock']].append(item)

    df = pd.read_csv("stock_details_5_years.csv")
    print(df.shape,df.head(2))

    with open("stock_details_5_years.csv", mode='r',newline='') as file:
        reader = csv.DictReader(file)

        for row in reader:
            item = dict(row)
            if row["Company"] not in price_mapping: # check if ticker already in mapping and add
                price_mapping[row["Company"]] = {} # eg: TSLA:{}
            Date = str(str(row["Date"]).split(" ",1)[0])
            price_mapping[row["Company"]][Date] = item # TSLA:{data:{...}}
            # print(price_mapping[row["Company"]][Date])
    ########## Ensure Ticker compatibility for both sets, bidirectional relationship

    print("Before:",len(news_mapping),len(price_mapping))


    for ticker in news_mapping.copy():
        if ticker not in price_mapping:
            news_mapping.pop(ticker)

    for ticker in price_mapping.copy():
        if ticker not in news_mapping:
            price_mapping.pop(ticker)
    
    print("After:",len(news_mapping),len(price_mapping))

    return (news_mapping, price_mapping)




def prep_data_for_tokenization(news_mapping, price_mapping):
    '''
    Input format:
    news_mapping = {
        <ticker>: {
            'index': '0', 
            'headline': 'Stocks That Hit 52-Week Highs On Friday', 
            'url': 'https://www.benzinga.com/news/20/06/16190091/stocks-that-hit-52-week-highs-on-friday', 
            'publisher': 'Benzinga Insights', 
            'date': '2020-06-05 10:30:54-04:00', 
            'stock': 'A'}
            }
        },
    price_mapping = {
        <ticker>: {
            <2020-02-19>:{
                'Date': 2018-11-29 00:00:00-05:00,
                'Open': 43.829761
                'High: 43.863354 
                'Low': 42.639594
                'Close': 43.083508
                'Volume': 167080000
                'Dividends: 0.0
                'Stock Splits': 0.0,
                'Company': AAPL
            }
        }

    Output: 
    
    news_array = [
    "headline:Credit Suisse Maintains Underperform on Southern, Raises Price Target to $46
    url:https://www.benzinga.com/news/18/12/12808347/credit-suisse-maintains-underperform-on-southern-raises-price-target-to-46
    publisher:Vick Meyer
    date:2018-12-06
    day_of_week:Thursday
    ticker:SO
    price:37.8127479553223", ....
    ]

    target_array = [30.0,-1.23, 176.231, ....] -> Represents the relative percent change from ~current end of trading day to the next end of trading day
        
    Known issues with data
    price source: 2018-2023
    news source 2009-2020
    Currently we only have about 3 years worth of overlap
    '''
    news_array = []
    target_array = []
    for ticker in news_mapping:
        for news_object in news_mapping[ticker]: # filters through all news regarding a single ticker 
            single_news_item = ""

            headline = news_object["headline"]
            url = news_object["url"]
            publisher = news_object["publisher"]
            date_ = datetime.strptime(str(str(news_object["date"]).split(" ",1)[0]), "%Y-%m-%d").date() # may want to get day of week, news on friday vs monday may be different 
            day_of_week_int =  date_.isoweekday() # Monday is 1 and Sunday is 7
            day_of_week_string =  date_.strftime("%A")
            stock_ticker = news_object["stock"]
            before_date = date_ - timedelta(days=2)
            after_date  = date_ + timedelta(days=2)

            # Weekend fallback
            if before_date.weekday() > 4:  # Sat/Sun
                before_date -= timedelta(days=before_date.weekday() - 4)  # Friday

            if after_date.weekday() > 4:
                after_date += timedelta(days=7 - after_date.weekday())  # Monday

            before_str = before_date.strftime('%Y-%m-%d')
            after_str  = after_date.strftime('%Y-%m-%d')

            if before_str in price_mapping[ticker] and after_str in price_mapping[ticker]:
                price_before = float(price_mapping[ticker][before_str]["Close"])
                price_after = float(price_mapping[ticker][after_str]["Close"])
                target_percent_change = ((price_after - price_before) / price_before) * 100
                # Label it, use it
            else:
                continue  # Skip this sample

            # if end_of_day_price == None:
            #     continue
            # if end_of_next_trading_day_price == None:
            #     continue

            # target_percent_change = ((float(end_of_next_trading_day_price) - float(end_of_day_price)) / float(end_of_day_price)) * 100
            abs_change = abs(target_percent_change)
            if abs_change < 1.0: label = 0 # less than 1 percent : low volatility news
            elif abs_change < 3.0: label = 1 # 1->3 percent : medium volatility news
            else: label = 2 #: high volatility news, above 3 percent change 
            target_array.append(label)

            keys = ["headline","publisher", "date", "day_of_week", "ticker", "price"]
            values = [headline,publisher,date_.strftime("%Y-%m-%d"),day_of_week_string,stock_ticker,str(price_before)]
            for index, value in enumerate(values):
                formatted_item = f"{keys[index]}:{value} [SEP] "
                single_news_item += formatted_item
            news_array.append(single_news_item)

    return news_array, target_array
    # print("News_array_len", len(news_array))
    
    # for i, value in enumerate(news_array):
    #     print(value)



class StockNewsDataset(Dataset):
    def __init__(self, tokenized_inputs, targets):
        self.input_ids = tokenized_inputs["input_ids"]
        self.attention_mask = tokenized_inputs["attention_mask"]
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.targets[idx], dtype=torch.float)
        }





def training_data(max_length = 128):
    '''
    Input: 
    max_length = 128 : Used to change the bert encoder max length if you felt that adding more data was needed
        - Possible data to add is the days volume or other metrics
    
    data None -> The file parsing done for a specific file so no inputs 
    
    outputs:
    
    (
    tokenized_values: {input_ids:[[]..[]], token_type_ids:[[]..[]] ,attention_mask:[[]..[]]},

    target_percent_change []
    )
    '''
    (news, prices) = load_data()
    news_array, target_array = prep_data_for_tokenization(news,prices)
    tokenized_news_inputs = tokenize_inputs(news_array, max_length)
    return tokenized_news_inputs,target_array



# (news, prices) = load_data()
# news_array, target_array = prep_data_for_tokenization(news,prices)
# tokenized_news_inputs = tokenize_inputs(news_array)
# for outputs in tokenized_news_inputs:
#     print(outputs)
# tokenized_inputs, attention_mask = tokenized_news_inputs.split()


# df = pd.DataFrame(tokenized_inputs)

# df.head(4)



'''
The data:
Input: 
    - Ticker symbol
    - Headline
    - Date
    - Possibly Current price

Input Structure:
    Ticker: TSLA
    Date of News: 10/12/2026
    Day of Week: Friday
    News Headline: "Tesla Announces record breaking Q3, 200% increase in revenue"
Output Structure : Percent change: +4%

Output:
    - Next percentage from day news was released to end of next trading day 
        - Ex: Input:  "TSLA, 12/03/2024, "Tesla seems to have solved AGI" 
              Output: 14%

Data gather: 


Stock ticker  

'''