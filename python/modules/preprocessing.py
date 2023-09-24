from modules.GlobalVariables import *

import pandas as pd
import numpy as np
import re
import swifter
import spacy
nlp = spacy.load("en_core_web_sm")


class PreprocessFilter:
    def __init__(self):
        self = self
    
    def filter_column(self, run):
        if run:
            # Load data
            data = LoadData().read_data("00.raw_patent_v2.xlsx", "csv2023-08-23-11-37-26")
            # Filter columns
            filter_data = data[FilterData().filter_columns]
            filter_data.columns = FilterData().filter_columns_renamed
            filter_data["year"] = filter_data["year"].astype(int)
            filter_data.to_pickle("data\\01.subset_filterColumn.pkl")
            return filter_data
        
    def filter_row(self, run):
        if run:
            data = LoadData().read_data("01.subset_filterColumn.pkl")
            # Exclude papers which include nano* but not nano-related // only select semiconductor-related papers
            data2 = data.dropna()
            data2["corpus"] = data2["title"].astype(str) + " " + data2["abstract"].astype(str) + " " + data2["claims"].astype(str)
            data2["country"] = data2["country"].astype(str) # Convert the "nationality" column to string type
            # Select sample                
            filtered_df = self._select_sample(data2, "country", FilterData().region_list)
            filtered_df.to_pickle("data\\01.subset_filterRow.pkl")
            return filtered_df

        
    def _select_sample(self, data, column_, sample_):
        if isinstance(sample_[0], str):
            pattern = "|".join(sample_)
            subset_data = data[data[column_].str.contains(pattern, case=False, na=True)]    
        elif isinstance(sample_[0], int):
            sample_string = [str(x) for x in sample_]
            pattern = "|".join(sample_string)        
            subset_data = data[data[column_].astype(str).str.contains(pattern, case=False, na=True)]            
        return subset_data


class PreprocessCleanse:
    def __init__(self):
        self = self
    
    def cleanse_text(self, run):
        if run:
            data = LoadData().read_data("01.subset_filterRow.pkl")
            data["corpus_cleansed"] = data["corpus"].swifter.apply(lambda x: self._cleansing(x))
            # subset_df = data[data['corpus_cleansed'].str.contains(r'\belectr\b')]
            # test_mask = data['corpus_cleansed'].str.contains('electr', case=False, na=False)
            # subset_df = data[test_mask]

            data.to_pickle("data\\02.text_cleansed.pkl")

    def data_for_stm(self, run):
        if run:           
            data = LoadData().read_data("02.text_cleansed.pkl")
            # descriptive trend
            # Create a pivot table to count publications for each year and country
            df_trend = data.pivot_table(index='year', columns='country', aggfunc='size', fill_value=0)
            # Reset index to have 'year' as a column
            df_trend = df_trend.reset_index()
            # Rename columns for clarity
            df_trend.columns.name = None
            df_trend.rename(columns={'CN': 'Chinese', 'JP': 'Japanese'}, inplace=True)
            df_trend.to_excel("data\\03.trend.excel.xlsx", index=False)

            # make data suitable to run STM
            # to have a full data set with no missing data
            year_range = data["year"].unique()
            country_values = data["country"].unique()
            complete_combinations = pd.DataFrame([(year, country) for year in year_range for country in country_values], columns=["year", "country"])
            data = complete_combinations.merge(data, on=["year", "country"], how="left").fillna({"corpus_cleansed": ""})
            data_sorted = data.sort_values(by=["country", "year"])
            # assign ID
            data_sorted["id"] = data_sorted["country"] + '-' + (data_sorted.groupby("country").cumcount() + 1).astype(str)
            data_sorted2 = data_sorted[["id", "year", "country", "corpus_cleansed"]]
            data_sorted2.dropna(inplace=True)
            data_sorted2.to_pickle("data\\03.stm_data.pkl")

            return data_sorted2
    
    def _cleansing(self, data):
        data_origin = data
        data = data.lower()
        # remove words in parantheses
        data = re.sub(RegEx().pattern_within_parentheses, "", data)
        # remove all punctuations
        data = re.sub(RegEx().pattern_punctuation, "", data)
        # remove stopwords
        data = re.sub(RegEx().stopwords, "", data)
        # data = self._extract_pos(data)
        # only keep english
        data = "".join(RegEx().pattern_english.findall(data))
        # replace all whitespaces into a single space
        data = re.sub(RegEx().pattern_whitepaces, " ", data)        
        data = " ".join([word for word in data.split() if len(word) != 1])
        return data
    
    def _extract_pos(self, data):
        try:
            text_nlp = nlp(data)
            extracted_text = " ".join([token.lemma_ for token in text_nlp if bool(re.match(RegEx().pat_pos, token.tag_)) == True])
        except ValueError: 
            extracted_text = ""
        return extracted_text        





class UnifyNames:
    def __init__(self, doc_type:str):
        self.doc_type = doc_type
    
    # Unify countries' names
    def unify_nationality(self, run):
        if run:
            if self.doc_type == "paper":
                data = LoadData().read_data(f"01.{self.doc_type}_filterRow.xlsx")
                data["nationality_cleansed"] = data["nationality"].swifter.apply(lambda x: self._mapLookup(x, ",", "nationality"))
                # check the affiliations' names
                affiliation_sorted = self._unique_list(data, "nationality_cleansed", ",")
                data.to_excel(f"data\\02.{self.doc_type}_unifyNames_countries.xlsx", index=False)
                return data             


    # unify affiliations' names
    def unify_affil(self, run):
        if run:
            data = LoadData().read_data(f"03.{self.doc_type}_collab.xlsx")
            data = data[["key","year", "keywords", "affiliations", "nationality_cleansed", "collab"]]
            data = data.dropna(subset=["affiliations"])
            data["affiliations"] = data["affiliations"].str.upper()
            # data=data.iloc[380:400]
            # unify affiliations' names by cleansing
            data["affiliation_cleansed"] = data["affiliations"].swifter.apply(lambda x: self._cleanse_name(x, ";"))
            # unify affilations' names by mapping
            data["affiliation_cleansed"] = data["affiliation_cleansed"].swifter.apply(lambda x: self._mapLookup(x, ";", "affil"))
            # check the affiliations' names
            self._unique_list(data, "affiliation_cleansed", ";")
            data.to_excel(f"data\\04.{self.doc_type}_unifyNames_affil.xlsx", index=False)
            return data

    def _cleanse_name(self, data, delimiter) -> str:
        words_to_remove_ = LoadData().read_data("99.look_up.xlsx", sheet_="exclusion")
        words_to_remove = words_to_remove_["exclusion"].astype(str).tolist()      
        list_ = []  # Provide a default value for the 'list_' variable to solve UnboundLocalError: local variable 'list_' referenced before assignment
        if isinstance(data, str) and data.startswith('['):
            data = re.sub(RegEx().pattern_square_brackets_quotation , "", data)
            list
            _ = data.split(delimiter)
        elif isinstance(data, str):
            list_ = data.split(delimiter)
        x_list2 = [x.strip() for x in list_] # remove unnecessary whitespaces
        unified_list = []
        for element in x_list2:    
            # Remove words within parantheses
            element_cleansed = re.sub(RegEx().pattern_within_parentheses, "", element)
            # Remove whitespace after a comma, ex) test, co >> test,co
            element_cleansed = re.sub(RegEx().pattern_after_whitespace, "", element_cleansed)
            # Remove commas and periods
            element_cleansed = re.sub(r"[.,]","", element_cleansed)
            # Remove unnecessary words such as corporation, llc, inc, etc.
            element_cleansed = ' '.join([word for word in element_cleansed.split() if word not in words_to_remove])
            element_cleansed.strip()
            unified_list.append(element_cleansed)
        return delimiter.join(unified_list)        


    def _mapLookup(self, data, delimiter, type_): # type_ = nationality / affil
        lookup_table = LoadData().read_data(f"99.look_up.xlsx", sheet_ = f"{self.doc_type}_{type_}")
        list_ = []
        if isinstance(data, str) and data.startswith('['):
            data = re.sub(RegEx().pattern_square_brackets_quotation , "", data)
            list_ = data.split(delimiter)
        elif isinstance(data, str):
            list_ = data.split(delimiter)            
        x_list2 = [x.strip() for x in list_] # remove unnecessary whitespaces
        unified_list = []
        for element in x_list2:
            if type_ == "nationality":
                # remove all punctuations
                element = re.sub(RegEx().pattern_punctuation, "", element)            
                # remove all strings before "usa" to unify names such as "nc usa", "pa usa", "tn usa", "mi usa", etc.
                if re.match(RegEx().pattern_excluding_except_us, element):
                    element = "usa"
            else: # type_ == "affiliations"
                # replace all whitespaces into a single space
                element = re.sub(RegEx().pattern_whitepaces, " ", element)
                # remove a word (1) if it is after a hyphen and (2) if it is the last word
                element = re.sub(RegEx().pattern_after_hyphen, "", element)
                # remove all whitepaces before and after any punctuation
                element = re.sub(RegEx().pattern_around_punctuation, r"\1", element)                
            # unify names by matching with lookup table
            lookup = lookup_table[lookup_table['before'] == element]['after']
            if not lookup.empty:
                unified_list.append(lookup.iloc[0])
            else:
                unified_list.append(element)
        unified_list = [x.strip() for x in unified_list]
        return delimiter.join(unified_list)

        

    def _unique_list(self, data, column_, delimiter):
        affiliations_list = data[column_].str.split(delimiter).tolist()
        # Flatten the list of affiliations
        flat_affiliations_list = [applicant for sublist in affiliations_list for applicant in sublist]
        flat_affiliations_list = [x.strip() for x in flat_affiliations_list]
        # Get the unique list of affiliations and their frequencies
        affiliations_freq = pd.Series(flat_affiliations_list).value_counts()
        # Sort the affiliations based on their frequency in descending order
        sorted_affiliations_freq = affiliations_freq.sort_values(ascending=False)
        df_sorted_affiliations_freq = pd.DataFrame({column_: sorted_affiliations_freq.index, 'freq': sorted_affiliations_freq.values})
        df_sorted_affiliations_freq.to_excel(f"data\\99.{self.doc_type}_subset_freq.xlsx", index=False)      

class Descriptive:
    def __init__(self, doc_type:str):
        self.doc_type = doc_type

    # Unify countries' names
    def annual_trend(self, type_, run):
        if run:
            if self.doc_type == "paper":
                data = LoadData().read_data(f"02.{self.doc_type}_unifyNames_countries.xlsx")
                data = data.dropna(subset=["affiliations"])
                year_range = data["year"].unique()
                data["collab"] = data["affiliations"].swifter.apply(lambda x: self._determine_collab(x, ";"))
                data = data[["key", "title", "year", "abstract", "keywords", "affiliations", "nationality_cleansed", "collab"]]
                data.to_excel(f"data\\03.{self.doc_type}_collab.xlsx", index=False)
                # annual trend
                if type_["overall"]:                    
                    trend_overall = self._descriptive_annual_trend(data, year_range)                
                if type_["collab"]:
                    sub_data = data[data["collab"].str.contains("collab")]
                    trend_collab = self._descriptive_annual_trend(sub_data, year_range)                    
                with pd.ExcelWriter(f"data\\10.{self.doc_type}_trend.xlsx") as writer:
                    trend_overall.to_excel(writer, sheet_name="overall", index=False)
                    trend_collab.to_excel(writer, sheet_name="collab", index=False)

    def _determine_collab(self, data, delimiter):
        if isinstance(data, str):
            if delimiter in data:
                return "collab"
            else: 
                return "sole"
        elif isinstance(data, str):
            if len(data) > 1:
                return "collab"
            else: 
                return "sole"
            

    def _descriptive_annual_trend(self, data, year_range):        
        trend_ = []
        for region in FilterData().region_list:
            sub_data = data[data["nationality_cleansed"].str.contains(region)]
            annual_trend = sub_data.groupby("year").size().reindex(year_range, fill_value=0).reset_index()
            annual_trend.columns = ["year", "trend"]
            annual_trend["region"] = region
            annual_trend = annual_trend.sort_values(by='year', ascending=True)
            trend_.append(annual_trend)
        merged_trend = pd.concat(trend_, ignore_index=True)
        return merged_trend