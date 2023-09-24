import pandas as pd
import re
import os
from os.path import join
import swifter

import multiprocessing
from multiprocessing import Pool, cpu_count
from multiprocessing import Manager
from tqdm import tqdm
import numpy as np
from functools import partial
from spacy.lang.en.stop_words import STOP_WORDS


class FilterData:
    def __init__(self):
        self.filter_columns = ["Application Number", "Application Year", "Application Country/Region",
                               "Title - DWPI", "Abstract", "Claims",
                               "IPC - Current", "Assignee/Applicant"]

        self.filter_columns_renamed = ["app_no", "year", "country",
                                       "title", "abstract", "claims",
                                       "ipc", "applicants"]
        
        self.region_list = ["JP", "CN"]
        self.period_list = [(2000, 2006), (2006, 2011), (2011, 2016), (2016, 2023)]
    
    def select_period(self, start_year, end_year):
        return list(range(start_year, end_year))


class RegEx:
    def __init__(self):
        # remove (1) values within parantheses, (2) parantheses, (3) whitespace before and after parantheses
        self.pattern_within_parentheses = r"\s*\([^)]*\)\s*"
        self.pattern_after_whitespace = r",\s+"

        self.pattern_select_square_brackets = r'\[|\]' # select square brackets
        self.pattern_square_brackets_quotation = r'[\[\]\'"]' # select square brackets and quotations(', ")
        self.pattern_punctuation = r'[^\w\s]' # select all punctuations
        self.pattern_punctuation_selected = r"[\-\/\(\)\;\.\,\&]"
        self.pattern_splitting_authors = r'\[(.*?)\]' # selecting values in between square brackets
        self.pattern_splitting_affiliations = r'\s*;\s*(?=\[)'  # split authors' information >> [...]...; [...]...;
        self.pattern_excluding_except_us = r'^[a-zA-Z]{2}.*usa$'
        self.pattern_around_punctuation = r'\s*([^\w\s])\s*'
        self.pattern_after_hyphen = r'-(\s*\w+)\s*\Z'
        self.pattern_whitepaces = r'\s+'
        self.pattern_around_and = r"\s*\&\s*"
        
        self.pattern_english = re.compile(r'[\x00-\x7F]+')  # Matches ASCII characters (English)

        new_stop_words = ["invention", "research", "1st", "recent", "paper", "technology", "new", "review", "future", "approach",
                          "claims", "claim", "batteries", "battery", "materials", "material", "percentage", "solid", "solidst", "said"]
        STOP_WORDS.update(new_stop_words)
        self.stopwords = re.compile(r"|".join(list(map(lambda x: r"\b"+rf"{x}"+r"\b", STOP_WORDS))))

        self.pat_pos = re.compile(r"NN[S|P|PS]{0,}|JJ[R|S]{0,}")


class LoadData:
    def __init__(self):
        self = self   

    def read_data(self, filename, sheet_="Sheet1"):
        default_path = os.getcwd()
        input_path = join(default_path, "data")        
        # change default directory to read data
        os.chdir(input_path)
        # read excel file
        if filename.endswith("xlsx"):
            data = pd.read_excel(filename, engine="openpyxl", sheet_name = sheet_, skiprows=2)
        # read pickle file
        else:
            data = pd.read_pickle(filename)        
        # reset default directory
        os.chdir(default_path)
        return data