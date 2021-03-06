from GoogleNews import GoogleNews
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, date

date_format = "%m/%d/%Y"


def gen_cal_dates(start_date, end_date):
    ''' Generate a list of dates based on start and end, to be fed to API '''

    delta = end_date - start_date
    datetime_list  = pd.date_range(end = end_date, periods = delta.days+1).to_pydatetime().tolist()
    
    return datetime_list
    

def googlenews_extract(date_range, num_pages, search_text):

    ''' Use googlenews package to extract top 30 stories per day based on search string '''
    
    df_days = []
    
    # loop through date range to ensure equal sample size from each day
    #TODO: if we want to pull multiple years of data, perhaps add multi-threading...not necessary for < ~20 calls
    for date in date_range:
        
        result = []
        googlenews = GoogleNews(start=date, end=date)
        googlenews.search(search_text)
        print("Search Date = ", date)
        
        for i in range(0, num_pages):

            print('Executing GoogleNews call #', i+1)

            googlenews.getpage(i)
            result_next = googlenews.result()
            print("Total records returned: ", len(result_next))
            
            df = pd.DataFrame(result_next)   
            df['date_calendar'] = date
        
        df_days.append(df) 
        appended_data = pd.concat(df_days)

    df_news = appended_data.reset_index(drop=True).drop(['date'], axis=1)
      
    return df_news


def sentiment_scores(df, field):

    ''' Evaluate News titles with NLTK sentiment analyzer '''

    analyzer = SentimentIntensityAnalyzer()

    scores = df[field].apply(analyzer.polarity_scores).tolist()
    df_scores = pd.DataFrame(scores)

    df_scored = df.join(df_scores, rsuffix='_right')

    return df_scored


def avg_daily_sentiment(df, date_field, measure):

    ''' Compute daily average sentiment across daily headlines '''

    avg_sent = df.groupby([date_field])[measure].mean()
    avg_sent_df = pd.DataFrame(avg_sent)
    avg_sent_df = avg_sent_df.rename(columns={measure: 'avg_sentiment'})
    
    return avg_sent_df
    

def read_csv(path, date_field, date_format):

    data = pd.read_csv(path, skiprows=0, parse_dates=[date_field])
    df = pd.DataFrame(data)

    # convert date to desired format
    df[date_field] = df[date_field].apply(lambda x: x.strftime(date_format))
    
    return df



def main():
    
    # generate dates for all of 2020 through present
    #datetime_list = gen_cal_dates(date(2020, 1, 1), date.today())
    datetime_list = gen_cal_dates(date(2020, 1, 1), date(2020, 1, 5))

    # re-format dates
    #TODO use list comprehension instead
    stringdate_list = []
    for i in range(len(datetime_list)):
        format_date = datetime.strftime(datetime_list[i], date_format)
        stringdate_list.append(format_date)
        
    # convert min and max date to hyphenated for to_csv usage
    min_date = stringdate_list[0] 
    max_date = stringdate_list[-1]
    min_date = min_date.replace("/", "-")
    max_date = max_date.replace("/", "-")

    # pull headlines for specified date range
    df_news = googlenews_extract(stringdate_list, 2, 'bitcoin')

    # sentiment for all headlines
    df_news_scored = sentiment_scores(df_news, 'title')

    # calc avg sentiment by day
    date_avg_sent_df = avg_daily_sentiment(df_news_scored, 'date_calendar', 'compound')

    # read bitcoin prices
    bitcoin_df = read_csv("data/df_with_leads.csv", 'date', 'date_format')

    # join news scores to bitcoin prices
    prices_join_news_scores = bitcoin_df.merge(date_avg_sent_df, left_on='date', right_on='date_calendar', how='left')
    prices_join_news_scores.columns

    # write joined results
    file_name = f"Bitcoin Prices with GoogleNews Avg Sentiment for '{min_date}'-'{max_date}'"
    prices_join_news_scores.to_csv("data/"+file_name+".csv", sep=',', index=False)
    print (f"'{file_name}.csv' written to '~/data/'...")

if __name__== "__main__" :
    main()