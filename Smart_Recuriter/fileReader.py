from operator import index
from Smart_Recuriter import Cleaner
from pandas import pandas as pd
import os
import glob
from Smart_Recuriter import Similar


def Algorithm(jobs, resumes):
    # job_description_names = os.listdir(job_desc_dir)
    # resume_names = os.listdir(resume_dir)

    # document = []
    # def read_resumes(list_of_resumes, resume_directory):
    #     placeholder = []
    #     path = resume_directory
    #     all_files = glob.glob(path + "/*.pdf")

    #     for i in range(len(all_files)):
    #        doc = fitz.open(all_files[i])  # open document
    #        text=''
    #        temp = []
    #        temp.append(all_files[i])
    #        for page in doc:  # iterate the document pages
    #            text = text + str(page.get_text())  # get plain text (is in UTF-8)
    #        tx=" ".join(text.split('\n'))
    #        temp.append(tx)
    #        if len(tx) > 5:
    #            placeholder.append(temp)
    #     return placeholder

    # document = read_resumes(resume_names, resume_dir)

    document = resumes

    def get_cleaned_words(document):
        for i in range(len(document)):
            raw = Cleaner.Cleaner(document[i][1])
            document[i].append(" ".join(raw[0]))
        return document

    Doc = get_cleaned_words(document)

    Database = pd.DataFrame(Doc, columns=["Name", "Context", "Cleaned"])

    # Database.to_csv("Resume_Data.csv", index=False)

    # Database.to_json("Resume_Data.json", index=False)

    # def read_jobdescriptions(job_description_names, job_desc_dir):
    #     placeholder = []
    #     path = job_desc_dir
    #     all_files = glob.glob(path + "/*.pdf")

    #     for i in range(len(all_files)):
    #        doc = fitz.open(all_files[i])  # open document
    #        text=''
    #        temp = []
    #        temp.append(all_files[i])
    #        for page in doc:  # iterate the document pages
    #            text = text + str(page.get_text())  # get plain text (is in UTF-8)
    #        tx=" ".join(text.split('\n'))
    #        temp.append(tx)
    #        if len(tx) > 5:
    #            placeholder.append(temp)
    #     return placeholder

    # job_document = read_jobdescriptions(job_description_names, job_desc_dir)

    job_document = jobs
    Jd = get_cleaned_words(job_document)

    jd_database = pd.DataFrame(Jd, columns=["Name", "Context", "Cleaned"])

    # jd_database.to_csv("Job_Data.csv", index=False)

    def calculate_scores(resumes, job_description):
        score = Similar.match(resumes['Cleaned'], job_description['Cleaned'])
        return score

    Database['Scores'] = calculate_scores(Database, jd_database)

    Ranked_resumes = Database.sort_values(
        by=['Scores'], ascending=False).reset_index(drop=True)

    Ranked_resumes['Rank'] = pd.DataFrame(
        [i for i in range(1, len(Ranked_resumes['Scores'])+1)])
    return Ranked_resumes

# result=Algorithm(resume_dir, job_desc_dir)
