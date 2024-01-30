import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from src.logger import get_console_logger
from src.path import DATA_DIR

logger = get_console_logger()

try:
    ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
    ALPACA_API_SECRET = os.environ["ALPACA_API_SECRET"]

except KeyError:
    logger.error(
        "Plase set the eviroment variabel ALPACA_API_KEY and ALPACA_API_SECRET "
    )    
    raise

@dataclass
class News:
    headline: str
    summary: str
    context: str
    date: datetime
    
def fect_batch_of_news(
    from_date: datetime,
    to_date: datetime,
    page_token: Optional[str] = None,
) -> Tuple[list[News], str]:
    """_summary_

    Args:
        from_date (datetime): _description_
        to_date (datetime): _description_
        pate_token (Optional[str], optional): _description_. Defaults to None.

    Returns:
        Tuple[list[News], str]: _description_
    """
    # prepare the request URL
    headers = {
        "Apca-Api-Key-Id": ALPACA_API_KEY,
        "Apca-Api-Secret-Key": ALPACA_API_SECRET,
    } 
    params = {
        "start": from_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": to_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 50,
        "include_contect": True,
        "sort": "ASC"
    }  
    if page_token is not None:
        params["page_token"] = page_token
    
    url = "https://data.alpaca.markets/v1beta1/news"  
    
    # ping API
    response = requests.get(url, headers=headers, params=params)
    
    # parse output
    list_of_news = []
    next_page_token = None
    if response.status_code == 200:
        # Parse respone into json
        news_json = response.json()
        
        # extract next page token (if any)
        next_page_token = news_json.get("next_page_toke", None)
        
        for n in news_json["news"]:
            list_of_news.append(
                News(
                    headline=["headline"],
                    date=["updated_at"],
                    summary=["summary"],
                    context=["context"]
                )
            ) 
    else:
        logger.error("Request failed with status code:", response.status_code)
    return list_of_news, next_page_token

def save_news_to_json(news_list: List[News], filename: Path):
    news_data = [
        {
            "headline": news.headline,
            "date": news.date,
            "summary": news.summary,
            "context": news.context
        }
        for news in news_list
    ]
    with open(filename, "w") as json_file:
        json.dump(news_data, json_file, indent=4)
        
def download_historical_news(
    from_date: datetime,
    to_date: datetime,
) -> Path:
    """
    downlods hitorical news from Alpaca api and stores the in a file located at 
    the path returned by this funtion
    """ 
    
    logger.info(f"Download historical news from {from_date} to {to_date}")
    list_of_news,  next_page_token = fect_batch_of_news(from_date, to_date)
    logger.info(f"Fecdhed {len(list_of_news)} news")
    logger.debug(f"Next page token")
    
    while next_page_token is not None:
        batch_of_news, next_page_token = fect_batch_of_news(
            from_date, to_date, next_page_token
        )
        list_of_news += batch_of_news
        logger.info(f"Fethed a total of {len(list_of_news)}")
        logger.debug(f"Next page toke {next_page_token}")
        logger.debug(f"Last date in batch: {batch_of_news[-1].date}")
        
    # save to file
    path_to_file = (
        DATA_DIR
        / f'news_{from_date.strftime("%Y-%m-%d")}_{to_date.strftime("%Y-%m-%d")}.json'
    )
    save_news_to_json(list_of_news, path_to_file)
    
    logger.info(f"News data saved to {path_to_file}")
    
    # return path to file
    return path_to_file   
        
       