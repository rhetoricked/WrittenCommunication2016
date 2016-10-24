'''
GenderGenreMod2
Copyright 2016 Brian N. Larson and licensors

GENDER/GENRE PROJECT CODE: Module 2
This code is the second segment used to generate and analyze the data for the article
Gender/Genre: The Lack of Gendered Register in Texts Requiring Genre Knowledge.
_Written Communication_, 33(4), 360–384. https://doi.org/10.1177/0741088316667927
(the "Article"). If you use this code, you should cite to the Article.

This little program should take the XML clean text out of the XML files earlier
created in Python and move it into a corpus directory for use by NLTK.

It does not route output files to two subdirectories
categorized by gender; instead, it puts everything out to a single directory.

'''
######
#TOOLS AND OPERATING SYSTEM stuff
######

from __future__ import division
import sys
sys.path.append('/users/BrianLarson/terminal/PythonCode')

import numpy, re, pprint, matplotlib, pylab #re is for regular expressions
import nltk
from lxml import etree
import os
import csv
import shutil
import logging
import datetime
now = datetime.datetime.now().strftime("%y-%m-%d %H.%M.%S")
from io import StringIO #"For strings StringIO can be used like a file opened in
                        #text mode." See https://docs.python.org/3/library/io.html

# The following parser change for lxml results from this recommendation:
# http://lxml.de/FAQ.html#why-doesn-t-the-pretty-print-option-reformat-my-xml-output
# That recommendation related to Python 2.7.
parser = etree.XMLParser(remove_blank_text=True)
#it affects subsequent etree.parse calls that use it as a second argument

######
#DIRECTORY AND FILE VARIABLES
######
#These set some of the working directories and files. They are set manually,
#and users should be sure that the directories exist and are empty before running
#the code.

xmlFromPython_dir = "XMLOutfromPython/"

print ("\n \n \nYou must enter a directory path for a directory that contains ")
print ("a folder called " + xmlFromPython_dir + " with XML data files from Module 1.")
print ("Be sure to include the / at the end of the path! \n \n")
home_dir = input("Enter the path for the data working directory root:")
#The home_dir is the overall working directory for all four modules.

#The next line sets the location of files output from Module 1.
xmlinput_dir = home_dir + xmlFromPython_dir

#The next four lines set the output directors for this module.
#BNL should add lines that create these, for each path, use this:
#os.makedirs(path, exist_ok=True)
nltkcorpus_dir = home_dir + "NLTKCorporaUncatUntag/"
nltkfulltext_dir = nltkcorpus_dir + "Fulltext/"
nltkfacttext_dir = nltkcorpus_dir + "Facttext/"
nltknonfacttext_dir = nltkcorpus_dir + "Nonfacttext/"

#####
#DEBUGGING CODE
#####
#The following lines relate to options for debugging the code.

#The following line sets up debugging, which allows the recording of notices
#regarding the progress of the module to be printed to the screen or recorded
#in a log file.
logging_file = home_dir + 'Module2' + " " + str(now + ".log")
logging.basicConfig(filename=logging_file, filemode='w', level=logging.DEBUG)
#To log to a file, add these parameters to previous basicConfig:
# filename=logging_file, filemode='w',
#To log to the console, delete the parameters in the previous line.

#This code records some basic run information
logging.debug(" Gender/genre Module 2: Run " + str(now))
logging.debug(" Run on data in " + home_dir)
logging.debug(" Source of xml files from Module 1: " + xmlinput_dir)
logging.debug(" Output from this module in: " + nltkcorpus_dir)

#For testing, it may be desirable to pull just a few papers. The next variable works
#with code below to select only those files from the start_wd directory that
#begin with these characters. Note that these are strings and should be only
#four characters long.
sought_papers = ["1007", "1055", "2021"]
logging.debug(" sought_papers identified: " + str(sought_papers))

######
#FUNCTIONS SEGMENT
# sets up functions that will be used below.
######

#FUNCTION textcorpusout
# This function writes a corpus text file out to an appropriate directory based
# on the parameters it receives.
def textcorpusout(paper_num, text, outdir, seg_ID):
    out_doc_full_path = outdir + paper_num + seg_ID + ".txt"
    f = open(out_doc_full_path, encoding='utf-8', mode='w+')
    f.write(text)
    f.close()

#####
#MAIN LOOP
#####
# This loop iterates over files in xmlinput_dir directory and does stuff to files.

#Iterate through the files in the xmlinput_dir.
for orig_doc_name in os.listdir(xmlinput_dir):
    paper_num = orig_doc_name[0:4] #Note: this makes paper_num a str
    if (not orig_doc_name.startswith('.') ) :
        #If-statement screens out Mac OS hidden files, names of which start '.'
        #If you want only limited files, add this condition to the preceding
        #if-statement:
        #and paper_num in sought_papers
        #and then uses only files identified in sought_papers list above
        orig_doc_full_path = xmlinput_dir + orig_doc_name

        #Parse file with defined parser creating ElementTree
        gate_doc = etree.parse(orig_doc_full_path, parser)
        #Get root Element of parsed file
        doc_root = gate_doc.getroot()
        #Next two lines are for debugging.
        logging.debug("\n\n")
        logging.debug("Paper " + paper_num + " loaded.")

        #These lines grab the Analysis_Gender value from the text and assign
        #it to var gender. We'll use it to categorize the NLTK corpus
        #and to withhold non-gender-categorized texts from the corpus.
        #BNL: This prolly should be a function.
        gender = ""
        gg = doc_root.find("GG")
        quest = gg.find("Questionnaire")
        for i in quest.iter("Feature"):
            if i.get("Name") == "Analysis_Gender":
                gender = i.get("Value")

        #These lines save the files out to the appropriate location, if the
        #file is categorized for gender (some files are not)
        if gender in ["0", "1"]:
            #The next few lines extract the text segments.
            cleantext = doc_root.find("Cleantext")
            fulltext = cleantext.find("CleanFull").text
            facttext = cleantext.find("CleanFact").text
            nonfacttext = cleantext.find("CleanNonFact").text

            # These lines for debugging.
            logging.debug("Fulltext length: " + str(len(fulltext)))
            logging.debug("Facttext length: " + str(len(facttext)))
            logging.debug("Nonfacttext length: " + str(len(nonfacttext)))

            #The next two lines compare the lengths of the segments. Since
            #fulltext should be roughly equal in legnth to the other two
            #segments put together, it prints a warning to the console if that
            #is not the case.
            checklength = len(fulltext) - len(facttext) - len(nonfacttext)
            if abs(checklength) > 30:
                logging.debug(paper_num + ": CHECKLENGTHS! Difference = " + str(checklength))
                print(paper_num + ": CHECKLENGTHS! Difference = " + str(checklength))

            #The next three lines save out the corpora.
            logging.debug('Executing textcorpusout(paper_num, fulltext, nltkfulltext_dir, "Full")')
            textcorpusout(paper_num, fulltext, nltkfulltext_dir, "Full")
            logging.debug('Executing             textcorpusout(paper_num, facttext, nltkfacttext_dir, "Fact")')
            textcorpusout(paper_num, facttext, nltkfacttext_dir, "Fact")
            logging.debug('Executing             textcorpusout(paper_num, nonfacttext, nltknonfacttext_dir, "Nonfact")')
            textcorpusout(paper_num, nonfacttext, nltknonfacttext_dir, "Nonfact")

        else:
            logging.debug(paper_num + ": No Gender for Analysis! Not processed into corpus.")
            print(paper_num + ": No Gender for Analysis! Not processed into corpus.")
