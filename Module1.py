'''
GenderGenreMod1
Copyright 2016 Brian N. Larson and licensors

GENDER/GENRE PROJECT CODE: Module 1
This code is the first of several segments used to generate the data for the article
Gender/Genre: The Lack of Gendered Register in Texts Requiring Genre Knowledge.
_Written Communication_, 33(4), 360–384. https://doi.org/10.1177/0741088316667927
(the "Article"). If you use this code, you should cite to the Article.

This code takes XML output from GATE (Cunningham, Maynard & Bontcheva, 2011),
enriches it with data from the questionnaire that Larson administered to
students, produces clean texts of sections of the student documents after
removing citations, etc.

The resulting XML files can be used to create corpora for processing by NLTK,
which is work begun in Module2.

NOTE: As of April 5, 2016, this code fails to perform on a small number (7)
of the 200 or so texts put through it. Performance/responses are explained in
Evernote note available to the project team. The log file for this
program makes it possible to see where the problems are, and they can be
corrected manually before the next phase.

WORKS CITED
Cunningham, H., Maynard, D., & Bontcheva, K. (2011). Text Processing with GATE.
University of Sheffield Department of Computer Science.

'''

######
#TOOLS AND OPERATING SYSTEM stuff
######

from __future__ import division
import sys

import re #re is for regular expressions
import pprint #pretty print for formatting xml output to be more human-readable
import matplotlib
import pylab
import nltk
from lxml import etree
import os
import csv
import shutil
import codecs
import logging
import datetime
now = datetime.datetime.now().strftime("%y-%m-%d %H.%M.%S")
from io import StringIO #"For strings StringIO can be used like a file opened in
                        #text mode." See https://docs.python.org/3/library/io.html

# The following parser change for lxml results from this recommendation:
# http://lxml.de/FAQ.html#why-doesn-t-the-pretty-print-option-reformat-my-xml-output
# That recommendation related to Python 2.7.
parser = etree.XMLParser(remove_blank_text=True, encoding='UTF-8')
#it affects subsequent etree.parse calls that use it as a second argument


######
#DIRECTORY AND FILE VARIABLES
######
#These set some of the working directories and files. They are set manually,
#and users should be sure that the directories exist and are empty before running
#the code.

gate_source = "XMLoutfromGATE/"
print ("\n \n \nYou must enter a directory path for a directory that contains ")
print ("a folder called " + gate_source + " with XML data files from GATE.")
print ("Be sure to include the / at the end of the path! \n \n")
home_dir = input("Enter the path for the data working directory root:")
#The next directory is the subdirectory where the xml files from GATE are stored.
#This module does NOT alter the original files
start_wd = home_dir + gate_source
#The next folder is for files from GATE that do not have human annotations in them.
#They are regarded as "defective" for purposes of this code.
#They are copied to this folder to allow inspection.
defective_xml_out = home_dir + "DefectiveXMLfromGATE/"
#The next folder is where the output of this code is placed.
xml_output_dir = home_dir + "XMLOutfromPython/"
#The next line sets the file where the CSV data from the Excel export is. Obviously,
#the named file has to be in home_dir
csv_file = home_dir + 'MasterDataForXML.csv'

#####
#DEBUGGING CODE
#####
#The following lines relate to options for debugging the code.

#The following line sets up debugging, which allows the recording of notices
#regarding the progress of the module to be printed to the screen or recorded
#in a log file.
logging_file = home_dir + 'Module1' + " " + str(now + ".log")
logging.basicConfig(filename=logging_file, filemode='w', level=logging.DEBUG)
#To log to a file, add these parameters to previous basicConfig:
# filename=logging_file, filemode='w',
#To log to the console, delete the parameters in the previous line.

#This code records some basic run information
logging.debug(" Gender/genre Module 1: Run " + str(now))
logging.debug(" Run on data in " + home_dir)
logging.debug(" Source of GATE xml files: " + start_wd)
logging.debug(" Source of CSV data: " + csv_file)
logging.debug(" Output from this module in: " + xml_output_dir)

#For testing, it may be desirable to pull just a few papers. The next variable works
#with code below to select only those files from the start_wd directory that
#begin with these characters. Note that these are strings and should be only
#four characters long.
sought_papers = ["1007", "1055", "2021"]
logging.debug(" sought_papers identified: " + str(sought_papers))

#####
#OTHER RUNTIME OPTIONS
#####

#Large segments excluded from processing
#The files in GATE are manually annotated to delimit several large segment types.
#The next variable identifies the large segment types that are excluded from processing
#in this module.
lg_segments_out = ["Caption", "TOCTOA", "OtherFormal"]
logging.debug(" Large segments excluded: " + str(lg_segments_out))

#Small segments excluded from processing
#The files in GATE are manually annotated to delimit several small segment types.
#The next variable identifies the small segment types that are excluded from
#processing in this module.
sm_segments_out = ["Heading", "Footnote", "Cite", "Blockquote"]
logging.debug(" Small segments excluded: " + str(sm_segments_out))

#The next two variables ID the annotation sets that are possible based on who
#the annotators were. These are the initials of the human annotators who did
#annotations in GATE.
as1 = "SLL"
as2 = "BNL"
logging.debug(" Annotators identified: " + as1 + " and " + as2)

######
#FUNCTIONS SEGMENT
# sets up functions that will be used below.
######

#FUNCTION add_unq_subelement
# Adds a unique subelement to parent.
# parent is a tree element or subelement.
# name will be the name of a subelement under parent
def add_unq_subelement(parent, name):
    #First check to see if there is already an element with this name.
    name_present = False
    #name_present is switched to True only if an element by this name is already present.
    for element in parent: #Test whether element by this name is already present.
        if element.tag == name:
            logging.exception( "Func add_unq_subelement: This parent already has a " + name + " subelement!")
            name_present = True
    #If this subelement does not already exist, create it.
    if name_present == False:
        return etree.SubElement(parent, name)

#FUNCTION add_unq_feature
# Adds a unique feature under an element
# parent is a tree element or subelement. name and value will apply to the
# newly created feature
def add_unq_feature(parent, name, value):
    #First check to see if there is already a feature with this name.
    f_present = False
    #f_present is switched to True only if a Feature by this name is already
    #present.
    for i in range(len(parent)):
    #This loop iterates through the parent looking for a Feature subelement
    #with this name.
        if parent[i].tag == "Feature" and parent[i].get("Name") == name:
            logging.warning( "Func add_unq_feature: This element's " +
                name + " feature is already set!")
            f_present = True
    #If a Feature by this name does not already exist, create it.
    if f_present == False:
        return etree.SubElement(parent, "Feature", Name = name, Value = value)

#FUNCTION get_csv_data
# This function returns the record from the Micrsoft Excel worksheet
# (file_name) with information about the paper (paper_num).
# When this function is called, the code that calls it should test for FAIL;
# see the note after the return command below.
def get_csv_data(file_name, paper_num):
    logging.debug(" Running get_csv_data")
    with open(file_name, 'rU') as csvfile:
        csv_in = csv.DictReader(csvfile, dialect = 'excel')
        for record in csv_in:
            if record['UniqueID'] == paper_num:
                return record #This is a dictionary object.
            #If there is no match, this function returns "FAIL"
            #for which the line of code that calls it should test.

#FUNCTION add_xl_features
# This function takes an xml doc and adds GenderGenre elements and features.
# "GG" here stands for Gender/Genre, the name of this project.
def add_xl_features(docroot, paper_num, record):
    #Add GG element under root and Questionnaire under GG
    gg = add_unq_subelement(doc_root, "GG")
    add_unq_feature(gg, "PaperNum", paper_num)
    quest = add_unq_subelement(gg, "Questionnaire")
    for key in record.keys():
    #This loop uses the excel data pulled with the get_csv_data function.
        add_unq_feature(quest, key, record[key])
    logging.debug(" XL features added to " + paper_num)

#FUNCTION verify_annotation
# Examines an XML file to make sure an as1 or as2 annotation set appears on it.
# If there is none, this function copies the xml from GATE to a "defective"
# folder, making it easier for researcher to locate and inspect them.
def verify_annotation(docroot, paper_name):
    logging.debug(" Verify Annotation")
    names = [as1, as2]
    as_present = False
    #Tracking variable as_present is reset as true only if one of the
    #approved annotator sets is present.
    for i in range(len(docroot)):
        #This loop iterates over annotation sets and checks for presence of
        #one of the approved annotator sets.
        if docroot[i].tag == "AnnotationSet" and docroot[i].get("Name") in names:
            as_present = True
    if as_present == False:
        #Copies file to "defective" folder.
        paper_path = os.path.join(start_wd, paper_name)
        logging.warning(" NO ANNOTATIONS IN " + paper_name + ": Copying " + paper_path + " to " + defective_xml_out)
        shutil.copy(paper_path, defective_xml_out)
        return False
    else:
        return True

#FUNCTION extract_original_text
# This function extracts the original text TextWithNodes from the GATE output
# in string form. Opening the original file in regular file mode lets us get at
# the text in it and do REgex operations without freaking out the XML parser.
def extract_original_text(gatefile):
    logging.debug(" Running extract_original_text on " + gatefile)
    gatefile = start_wd + gatefile
    f = codecs.open(gatefile, encoding="utf-8")
    original = f.read()
    re_s = re.compile('<TextWithNodes>.*</TextWithNodes>', re.DOTALL)
    #In previous line, re.DOTALL option causes a '.' to match any character,
    #including a newline. Normally, '.' matches any character BUT a newline.

    result = re_s.findall(original)[0]
    f.close()
    return result

#FUNCTION get_annotation_set
# Given an xml root, this returns the identifier for the annotation set that
# should be used. It prefers the as1 set where both are present.
def get_annotation_set(root):
    logging.debug(" Running get_annotation_root")
    as1_present = False
    as2_present = False
    for i in range(len(root)):
        if root[i].tag == "AnnotationSet" and root[i].get("Name") == as1:
            as1_present = True
        else:
            if root[i].tag == "AnnotationSet" and root[i].get("Name") == as2:
                as2_present = True
    if as1_present:
        logging.debug(" Annotator as1 (" + as1 + ") is present.")
        return as1
    else:
        if as2_present:
            logging.debug(" Annotator as2 (" + as2 + ") is present.")
            return as2
        else:
            logging.warning(" Neither annotator as1 nor as2 is present.")
            return "ERROR: Neither annotator as1 nor as2 is present."

#FUNCTION extract_node_range
# This function works on string, not XML. It returns the node markers and all
# text between them indicated by the start and end nodes.
def extract_node_range(text,start,end):
    logging.debug(" Running extract_node_range")
    re_string = r'<Node id=\"' + start + r'\"/>.*<Node id=\"' + end + r'\"/>'
    u = re.compile(re_string, re.DOTALL)
    v = u.findall(text)
    if not v:
        logging.warning(" Node range not matched in this document!")
        return "ERROR: Node range not matched in this document!"
    else:
        return v[0]

#FUNCTION delete_span_text
# This function takes a string, not XML, that has node markers and text in it
# and removes all the text from between the specified start and end node
# markers, leaving the node markers.
def delete_span_text(text,start,end):
    logging.debug(" Running delete_span_text")
    re_pattern = r'<Node id=\"' + start + r'\"/>.*<Node id=\"' + end + r'\"/>'
    re_repl = '<Node id=\"' + start + '\"/><Node id=\"' + end + '\"/>'
    text = re.sub(re_pattern, re_repl, text, flags=re.DOTALL)
    return text

#FUNCTION delete_segments
# This function iterates through XML annotations. Edits happen to a text string.
# Function IDs segements where there is text that should be deleted, sending
# their start and end nodes to delete_span_text.
def delete_segments(text, root, aset, lg_segs, sm_segs):
    logging.debug(" Running delete_segments")
    for e in root.iter("AnnotationSet"):
        if e.get("Name") == aset:
            for f in e.iter("Annotation"):
                if f.get("Type") == "LargeSegment":
                    start_node = f.get("StartNode")
                    end_node = f.get("EndNode")
                    for g in f.iter("Value"):
                        if g.text in lg_segs:
                            text = delete_span_text(text,start_node,end_node)
                else:
                    if f.get("Type") in sm_segs:
                        start_node = f.get("StartNode")
                        end_node = f.get("EndNode")
                        text = delete_span_text(text,start_node,end_node)
    return text

#FUNCTION fact_delete
# Iterates through XML features. Identifies the start end end of the Fact
# section using XML features, and removes that text from the text string by
# sending to delete_span_text.
def fact_delete(text, root, aset):
    logging.debug(" Running fact_delete")
    for e in root.iter("AnnotationSet"):
        if e.get("Name") == aset:
            for f in e.iter("Annotation"):
                if f.get("Type") == "LargeSegment":
                    start_node = f.get("StartNode")
                    end_node = f.get("EndNode")
                    for g in f.iter("Value"):
                        if g.text == "Facts":
                            text = delete_span_text(text,start_node,end_node)
    return text

#FUNCTION lg_seg_xtract
# Originally, this is just to permit pulling the Facts section out, but it
# would work with other sections, too.
def lg_seg_xtract(text, root, aset, lg_seg):
    logging.debug(" Running lg_seg_xtract")
    for e in root.iter("AnnotationSet"):
        if e.get("Name") == aset:
            for f in e.iter("Annotation"):
                if f.get("Type") == "LargeSegment":
                    start_node = f.get("StartNode")
                    end_node = f.get("EndNode")
                    for g in f.iter("Value"):
                        if g.text == lg_seg:
                            re_string = r'<Node id=\"' + start_node + r'\"/>.*<Node id=\"' + end_node + r'\"/>'
                            u = re.compile(re_string, re.DOTALL)
                            try:
                                text = u.findall(text)[0]
                            except IndexError:
                                text = " Function error (lg_seg_xtract): Regex search function did not match any text."
                                logging.warning(text)
    return text

#FUNCTION nodes_out
# This function takes a string (not XML) and removes all the node markers in it!
def nodes_out(text):
    logging.debug(" Running nodes_out")
    re_pattern = r'<.*?>'
    re_repl = ''
    text = re.sub(re_pattern, re_repl, text, flags=re.DOTALL)
    return text

#FUNCTION cleanUTF8
# There is a tag in the header of the XML file that that comes from GATE.
# This function deletes that tag.
def cleanUTF8(doc):
    utf = r'encoding=.UTF-8.'
    found = re.search(utf, doc)
    logging.debug('__________________________________________')
    logging.debug(found)
    logging.debug('__________________________________________')
    cleanDoc = doc[0:found.start()] + doc[found.end():]
    return cleanDoc


#####
#MAIN LOOP
#####
# This loop iterates over files in start_wd directory and does stuff to files.

os.chdir(start_wd)

name_path = start_wd

#For loop iterates over all files in the directory with the GATE xml files.
for orig_gate_doc_name in os.listdir(name_path):
        #If statement screens out Mac OS hidden files, names of which start '.' and
        #researcher-excluded files, which start xxxx;
    if (not orig_gate_doc_name.startswith('.') and not orig_gate_doc_name.startswith('xxxx') ):
        #If sought_papers is specified above and researcher wants to limit run
        #to them, add the following condition to previous if statement
        #and orig_gate_doc_name[0:4] in sought_papers.

        orig_doc = open(name_path + orig_gate_doc_name, "r", encoding = "UTF-8")
        orig_doc_content = orig_doc.read() #This creates a string object.
        logging.debug("\n\n****************************** \nEntered Loop for " + orig_gate_doc_name)
        #logging.debug(orig_doc_content) #uncomment this only for really big log.

        #Next line removes the UTF8 tag from the XML file that came from GATE.
        orig_doc_content = cleanUTF8(orig_doc_content)
        #Next line parses file with defined parser creating ElementTree
        gate_doc = etree.parse(StringIO(orig_doc_content), parser)
        doc_root = gate_doc.getroot() #Get root Element of parsed file
        paper_num = orig_gate_doc_name[0:4] #NOTE: this makes paper_num a str

        #In next if statement, run verify_annotation as a condition of
        #processing the file further.
        #If verify_annotation is false, we move to next file.
        if verify_annotation(doc_root, orig_gate_doc_name):
            #The next line gets the corresponding data from CSV file.
            xl_rec_contents = get_csv_data(csv_file, paper_num)
            if xl_rec_contents == "FAIL":
                #If there's no Excel/CSV data matching, we move to the next file.
                logging.warning(" Paper num: " + paper_num + " not appearing in CSV file!")
            ##Assuming we pass those two checks, we get to process the file.

            ###########
            ##THIS IS THE PAYLOAD, WHERE EVERYTHING HAPPENS
            ###########
            else:
                logging.debug(" Paper passed tests; beginning processing.")

                #Add features from the Excel file (survey, etc.) to the xml file.
                add_xl_features(doc_root, paper_num, xl_rec_contents)

                #Save the results to a new XML file (we don't edit the
                #original from GATE at all.)
                #Next line creates new name for the xml output of this program
                #(this version replaced by next line)
                rev_gate_doc_name = xml_output_dir + orig_gate_doc_name

                gate_doc.write(rev_gate_doc_name, pretty_print = True,
                                xml_declaration=True, encoding='UTF-8')
                    #xml_declaration and encoding necessary to open UTF later
                logging.debug(" Paper saved as " + rev_gate_doc_name)

                #Next line reads the new file in. Parses the new file into etree.
                xml_doc = etree.parse(rev_gate_doc_name, parser)

                #Next line creates a variable for referring to the root of
                #this xml doc.
                xml_root = xml_doc.getroot()

                #Next line gets the TextWithNodes from the original file
                orig_w_nodes = extract_original_text(orig_gate_doc_name)

                #NOTE: In the next few lines, we are operating on the string
                #pulled in from orig_gate_doc_name, not on the
                #XML file. This allows us to treat the XML nodes as
                #strings rather than having to use complicated
                #XML parsing to clean them out of the string before
                #it can be passed to NLP functions like splitter, tokenizer, etc.

                #logging.debug(orig_w_nodes) #Uncomment this line for big log

                #The following lines delete from orig_w_nodes the text components
                #from all segments that will not be analyzed.
                #They prefer the annotation set ascribed to as1, otherwise use as2.
                ###
                gate_ann_set = get_annotation_set(xml_root)

                orig_w_nodes = (delete_segments(orig_w_nodes, xml_root,
                    gate_ann_set, lg_segments_out, sm_segments_out))

                #Next two lines create two new strings for the text that is
                #just the facts and text that is everything but facts.
                nonfact_w_nodes = (fact_delete(orig_w_nodes, xml_root,
                    gate_ann_set))
                facts_w_nodes = (lg_seg_xtract(orig_w_nodes, xml_root,
                    gate_ann_set, "Facts"))
                #print facts_w_nodes #Uncomment this line for debugging only.

                #We now have three strings, each cleansed of segments we don't
                #want but each having the node markers, which we now no longer
                #need. We first create a place in our XML document to hold the
                #results...
                cleantext = add_unq_subelement(xml_root,"Cleantext")
                cleanfull = add_unq_subelement(cleantext, "CleanFull")
                cleannonfact = add_unq_subelement(cleantext, "CleanNonFact")
                cleanfact = add_unq_subelement(cleantext, "CleanFact")

                #...then we put the results in after running through the
                #nodes_out function.
                try:
                    cleanfull.text = nodes_out(orig_w_nodes)
                except ValueError:
                    logging.warning(" XML error thrown while writing cleanfull.text: ")
                try:
                    cleannonfact.text = nodes_out(nonfact_w_nodes)
                except ValueError:
                    logging.warning(" XML error thrown while writing cleannonfact.text: ")
                try:
                    cleanfact.text = nodes_out(facts_w_nodes)
                except ValueError:
                    logging.warning(" XML error thrown while writing cleanfact.text: ")

                #Finally, we write the revised XML file out.
                xml_doc.write(rev_gate_doc_name, pretty_print = True,
                                xml_declaration=True, encoding='UTF-8')
