from modules.preprocessing import *
from modules.GlobalVariables import *
from modules.NetworkAnalysis import *

import datetime
import pytz

def main():
    # Print the current time in South Korea
    current_time_korea = datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    print("Started:", current_time_korea.strftime("%Y-%m-%d %H:%M"))

    run_ = {"run_filter_column": False, "run_filter_row": False, "cleanse_text":False, "data_for_stm":False
    , "doc_by_topics":False, "network_analysis":True}
    
    # 01. Filter out unnecessary columns and rows    
    preprocess_filter = PreprocessFilter()
    result = preprocess_filter.filter_column(run=run_["run_filter_column"])
    
    result = preprocess_filter.filter_row(run=run_["run_filter_row"]) # select samples

    # 02. Cleanse data
    result = PreprocessCleanse().cleanse_text(run=run_["cleanse_text"])

    # 03. Manipulate data for STM
    result = PreprocessCleanse().data_for_stm(run=run_["data_for_stm"])

    # STM 03. Merge data with the result of STM, doc_by_topics
    result = MergeData().doc_by_topics(run=run_["doc_by_topics"], no_topic = 10)

    # STM 07. Network Analysis
    result = NetworkAnalysis().knowledge_network(run=run_["network_analysis"], no_topic = 10)



    
    current_time_korea_finished = datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    print("Ended:", current_time_korea_finished.strftime("%Y-%m-%d %H:%M"))
    return result
    
if __name__ == "__main__":
    main()