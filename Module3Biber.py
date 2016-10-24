'''
GenderGenreMod3
Copyright 2016 Brian N. Larson and licensors

GENDER/GENRE PROJECT CODE: Module 3
This code is the third segment used to generate and analyze the data for the article
Gender/Genre: The Lack of Gendered Register in Texts Requiring Genre Knowledge.
_Written Communication_, 33(4), 360–384. https://doi.org/10.1177/0741088316667927
(the "Article"). If you use this code, you should cite to the Article.

This module tallies the instances of the features identified by Biber (1988,
1995) as constituting the involved-informational dimension in linguistic
register. It performs the analysis on all three versions of the corpus, Fulltext,
Facttext, and Nonfacttext. (The Article reported only on Fulltext.)

WORKS CITED
Biber, D. (1988). Variation across speech and writing. Cambridge  U.K.:
Cambridge University Press.
Biber, D. (1995). Dimensions of register variation : a cross-linguistic
comparison. Cambridge ;;New York: Cambridge University Press.

'''
#!/usr/bin/env python
from __future__ import division

#Set up run-time variables for files and directories.


import os
import shutil
import sys
import pickle
import logging
import datetime
now = datetime.datetime.now().strftime("%y-%m-%d %H.%M.%S")
from io import StringIO #"For strings StringIO can be used like a file opened in
                        #text mode." See https://docs.python.org/3/library/io.html
import nltk
import numpy, re, pprint, matplotlib, pylab #re is for regular expressions
import scipy.stats as stats
import csv


from lxml import etree
# This parser change for lxml results from this recommendation:
# http://lxml.de/FAQ.html#why-doesn-t-the-pretty-print-option-reformat-my-xml-output
parser = etree.XMLParser(remove_blank_text=True) #it affects subsequent etree.parse calls that use it as a second argument

#Sentence tokenizer (sentence splitter using sent_tokenize default, which is?)
from nltk.tokenize import sent_tokenize
#Word tokenizer Version using TreebankWorkTokenizer
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

######
#DIRECTORY AND FILE VARIABLES
######
#These set some of the working directories and files. They are set manually,
#and users should be sure that the directories exist and are empty before running
#the code.

#Following line is name selected for directory with corpora out from Module 2.
existingNLTK_dir = "NLTKCorporaUncatUntag/"

#Use next five lines only if you want to solicit end-user input regarding
#directorly
print ("\n \n \nYou must enter a directory path for a directory that contains ")
print ("a folder called " + existingNLTK_dir + " with corpora files from Module 2.")
print ("And THAT folder has to have a Pickles/ directory in it.")
print ("Be sure to include the / at the end of the path! \n \n")
home_dir = input("Enter the path for the data working directory root:")


run_root = home_dir

xml_dir = run_root + "XMLOutfromPython/"
pickle_dir = run_root + "NLTKCorporaUncatUntag/Pickles/"


#####
#DEBUGGING CODE
#####
#The following lines relate to options for debugging the code.

#The following line sets up debugging, which allows the recording of notices
#regarding the progress of the module to be printed to the screen or recorded
#in a log file.
logging_file = home_dir + 'Module3Biber' + " " + str(now + ".log")
logging.basicConfig(filename=logging_file, filemode='w', level=logging.DEBUG)
#To log to a file, add these parameters to previous basicConfig:
# filename=logging_file, filemode='w',
#To log to the console, delete the parameters in the previous line.

#This code records some basic run information
logging.debug(" Gender/genre Module 3Biber: Run " + str(now))
logging.debug(" Run on data in " + home_dir)
logging.debug(" Source of corpora files out from Module 2: " + existingNLTK_dir)
logging.debug(" Output from this module in: " + pickle_dir)

#For testing, it may be desirable to pull just a few papers. The next variable works
#with code below to select only those files from the start_wd directory that
#begin with these characters. Note that these are strings and should be only
#four characters long.
sought_papers = ["1007", "1055", "1116", "2013", "2021"]

#####
#WORD LISTS
#####
#The following lines load various wordlists used in this module.

with open((run_root + "PrivateVerbs.txt"), encoding = "utf-8") as f:
    privateVerbs = f.read().splitlines()
with open((run_root + "PubVerbs.txt"), encoding = "utf-8") as f:
    publicVerbs = f.read().splitlines()
with open((run_root + "SuaVerbs.txt"), encoding = "utf-8") as f:
    suasiveVerbs = f.read().splitlines()
priPubSuaVerbs = privateVerbs + publicVerbs + suasiveVerbs
with open((run_root + "Contractions.txt"), encoding = "utf-8") as f:
    contracts = f.read().splitlines()
with open((run_root + "AllP-Whp.txt"), encoding = "utf-8") as f:
    allP_whP = f.read().splitlines()
with open((run_root + "Amplifiers.txt"), encoding = "utf-8") as f:
    amplifiers = f.read().splitlines()
with open((run_root + "CL-P.txt"), encoding = "utf-8") as f:
    clpunct = f.read().splitlines()
allpunct = clpunct
allpunct.append(',')
with open((run_root + "Not.txt"), encoding = "utf-8") as f:
    analyticNeg = f.read().splitlines()
with open((run_root + "Dem.txt"), encoding = "utf-8") as f:
    demonstrative = f.read().splitlines()
with open((run_root + "FirstPers.txt"), encoding = "utf-8") as f:
    firstPersPro = f.read().splitlines()
with open((run_root + "IndefPro.txt"), encoding = "utf-8") as f:
    indefPro = f.read().splitlines()
with open((run_root + "SecPers.txt"), encoding = "utf-8") as f:
    secPersPro = f.read().splitlines()
with open((run_root + "Subpro.txt"), encoding = "utf-8") as f:
    subjectPro = f.read().splitlines()
with open((run_root + "WhoWords.txt"), encoding = "utf-8") as f:
    whoWords = f.read().splitlines()
with open((run_root + "WHPwords.txt"), encoding = "utf-8") as f:
    wHPWords = f.read().splitlines()

def process(section):
        nltkcorpus_dir = run_root + existingNLTK_dir + section + "/"
        logging_file = nltkcorpus_dir + "3BiberLog " + section + ".log" #5-1 moved out to global or deleted.
        results_file = pickle_dir + section + "Features.pickle"
        csv_out_file = run_root + "BiberOutput " + section + " " + str(now) + ".csv"

        #papers will be a list of dictionaries. Each dictionary will have the features for one paper.
        papers = [ ]

        def getgender(paper_num): #Move this function outside the process function.
                logging.debug(paper_num + " GET GENDER")
                logging.debug("\n_____")
                #Go into the XML document to get the gender.
                for f in os.listdir(xml_dir):
                    if f.startswith(paper_num):
                        xml_doc_full_path = xml_dir + f
                gate_doc = etree.parse(xml_doc_full_path, parser) #Parse file with defined parser creating ElementTree
                doc_root = gate_doc.getroot() #Get root Element of parsed file
                logging.debug("\n\n") # For debugging.
                logging.debug ("XML Paper " + paper_num + " loaded.") # This is just for debugging.
                #These lines grab the Analysis_Gender value from the text and assign it to var gender. We'll use it to categorize the NLTK corpus
                #and to withhold non-gender-categorized texts from the corpus. This prolly should be a function, but that will happen later.
                gender = ""
                gg = doc_root.find("GG")
                quest = gg.find("Questionnaire")
                for i in quest.iter("Feature"):
                    if i.get("Name") == "Analysis_Gender":
                        gender = i.get("Value")
                logging.debug("Gender: " + str(gender)) #for debugging
                return gender

        #Begin loop over papers
        #For each paper, we want to do these things:
        #Create a dictionary containing the following keys/values.
        #   1. Gender of text author.
        # See Biber 1995, p. 142, Table 6.1 for explanation of items.
        # See Reymann 2002 for discussion of implementation for automation.
        # Involved end:
        #   2. Count of private verbs: Tagged as verb and in list.
        #   3. Count of THAT-deletion:
        #       a. PUB/PRV/SUA + demonstrative pronoun/SUBJPRO5
        #       b. PUB/PRV/SUA + PRO/N + AUX/V
        #       c. PUB/PRV/SUA + ADJ/ADV/DET/POSSPRO + (ADJ) + N + AUX/V
        #       d. See lists of public, private, suasive verbs; dems; subpros
        #   4. Count of Contractions. See list.
        #   5. Count of Present-tense verbs. Use tagger tag VBP or VBZ
        #   6. Count of Second-person pronouns. See SecPers list.
        #   7. Count of DO as pro-verb. Any form of do (do does did done) NOT
        #       in the following:
        #       a. DO + (ADV) + V (DO as auxiliary)
        #       b. ALL-P/WHP + DO (where ALL-P/WHP is in file AllP-Whp.txt)
        #   8. Count of Analytic negation (with 'not' or 'n't')
        #   9. Count of demonstrative pronouns.
        #       that/this/these/those + V/AUX/WHP/and
        #  10. Count of General emphatics
        #       a. for sure
        #       b. a lot
        #       c. such a
        #       d. real + ADJ
        #       e. so + ADJ
        #       f. DO + V
        #       g. just
        #       h. really
        #       i. most
        #       j. more
        #  11. Count First-person pronouns. See FirstPers.txt.
        #  12. Count of Pronoun IT. Count all instances of it.
        #  13. Count of BE as main verb.
        #       Form of BE: be, is are, was, were, been--followed by any of these
        #           DT / PRP$ / IN / JJ / JJR / JJS
        #  14. Count of Causative subordination. Just count BECAUSE
        #  15. Count of Discourse particles. CL-P followed by any of:
        #       well / now / anyway / anyhow / anyways
        #  16. Count of Indefinite pronouns.
        #       a. any of IndefPro.txt OR
        #       b. no one
        #  17. Count of General hedges. Did not use all of Reymann, only
        #       at about / something like / more or less / almost / maybe
        #  18. Count of Amplifiers. If word is in Amplifiers.
        #  19. Count of Sentence relatives. Comma followed by which.
        #  20. Count of WH questions. Sentence boundary or CL-P + WhoWord.txt + MD
        #  21. Count of Possibility modals. All of can, could, may, and might
        #  22. Count of Non-phrasal coordination. Skipped here.
        #  23. Count of WH clauses. PUB/PRV/SUA + WHP/WHO + not(MD)
        #  24. Count of Final prepositions. IN followed by CL-P or comma.
        #  25. Count of (Adverbs). RB, RBR, RBS
        # Informational end:
        #  26. Count of Nouns.
        #  27. Mean Word Length. Mean length of all non-punct.
        #  28. Count of prepositions. IN, less 24. Did not exclude those that
        #       function as time and place adverbials, conjuncts or subordinators.
        #  29. Type-token ratio. Frequency count of types within first 400 words.
        #  30. Count of Attributive adjectives. JJ/JJR/JJS + (JJ/JJR/JJS or Noun)
        #  31. Count of (Place adverbials) Skipped here
        #  32. Count of (Agentless passives) Skipped here
        #  33. Count of (Past participial postnominal clauses) Skipped here.

        for file_name in os.listdir(nltkcorpus_dir):
            paper_num = file_name[0:4] #Note: this makes paper_num a str
            if ((not file_name.startswith('.')) ) : #Screens out Mac OS hidden files,
                                                #names of which start '.'
                #If you want only limited files, add this condition to the preceding if-statement: and paper_num in sought_papers
                #and then uses only files BNL
                #selected in sought_papers list above

                filepath = nltkcorpus_dir + file_name

                logging.debug("\n*************************************************************************")
                logging.debug ("LOADING " + filepath)
                logging.debug ("\n*************************************************************************")
                print("\n*************************************************************************")
                print ("LOADING " + filepath)
                print ("\n*************************************************************************")
                #opens the subject file , reads its contents, and closes it.
                f = open(filepath, encoding="utf-8")
                infile = f.read()
                f.close()

                #Declare the feature dictionary for this paper
                paper_dict = { }
                paper_dict["A_papernum"] = paper_num #NO-NORMALIZE

                #Populate the dictionary with key/0 for each feature in the list of features.
                paper_dict["A_tokens"] = 0 # Total tokens in paper NO-NORMALIZE
                paper_dict["A_sents"] = 0 #Total sentences in paper NO-NORMALIZE
                paper_dict["A_words"] = 0 #Total number of non-punct tokens. NO-NORMALIZE
                paper_dict["01Gender"] = getgender(paper_num) #NO-NORMALIZE
                paper_dict["02PrivateVerbs"] = 0 #Private verbs.
                paper_dict["03ThatDeletion"] = 0 #
                paper_dict["04Contractions"] = 0
                paper_dict["05PresVerbs"] = 0 #Present-tense verbs.
                paper_dict["06SecPersPrn"] = 0 #Second-person pronouns
                paper_dict["07DOproverb"] = 0 #Definition below.
                paper_dict["08AnalyticNeg"] = 0 #Negation with not or n't
                paper_dict["9DemoPron"] = 0 #Demonstrative pronouns
                paper_dict["10GenEmphatics"] = 0 #
                paper_dict["11FirstPersPrn"] = 0 #First-pe rson pronouns
                paper_dict["12It"] = 0 #count of IT
                paper_dict["13BeMain"] = 0 #BE as a main verb
                paper_dict["14CauseSub"] = 0 # Count because
                paper_dict["15DiscPart"] = 0 #Count as specified.
                paper_dict["16IndefPro"] = 0 # Indefinite pronouns
                paper_dict["17GenHedges"] = 0 #Count of general hedges
                paper_dict["18Amplifiers"] = 0
                paper_dict["19SentRelatives"] = 0 #Sentence followed by which.
                paper_dict["20WhQuestion"] = 0 # Wh questions.
                paper_dict["21PossModals"] = 0 # Possibility modals.
                paper_dict["22NonPhrasalCoord"] = 0 #Skipping.
                paper_dict["23WhClauses"] = 0 #WH clauses.
                paper_dict["24FinalPreps"] = 0 #Final prepositions.
                paper_dict["25Adverbs"] = 0
                paper_dict["26Nouns"] = 0
                paper_dict["27WordLength"] = 0 #mean length of non-punct words NO-NORMALIZE
                paper_dict["28Preps"] = 0 #Prepositions (other than 24)
                paper_dict["29TTRatio"] = 0 # Type-token ratio NO-NORMALIZE
                paper_dict["30AttribAdj"] = 0 #Attributive adjectives
                paper_dict["31PlaceAdverbs"] = 0 #Place adverbials. Skipped here.
                paper_dict["32AgentlessPass"] = 0 #Agentless passives. Skipped here.
                paper_dict["33PPPC"] = 0 #Past participial postnominal clauses. Skipped here.

                logging.debug("\n_____")
                logging.debug("Paper " + paper_num + "'s number of features: " + str(len(paper_dict)))
                logging.debug("\n_____")

                first400TTypes = []
                aggWordLength = 0
                #Tokenize infile into sentences. The result is a list of sentences.
                sentences = sent_tokenize(infile)
                paper_dict["A_sents"] = len(sentences)

                #Begin loop over sentences in paper.
                logging.debug("\n_____")
                logging.debug(paper_num + " LOOP OVER SENTENCES")
                logging.debug("\n_____")
                sentence_counter = 0
                for i in sentences: #For each sentence in the paper...
                    logging.debug("\nPaper-Sentence number: " + paper_num + "-" + str(sentence_counter))
                    logging.debug(i) # This is just for debug.
                    #Word-tokenize it.
                    tokenized = tokenizer.tokenize(i) #Result is a list of word-tokens.
                    logging.debug("\nTokenized sentence: " + str(sentence_counter))
                    logging.debug(tokenized) #for debug only

                    #POS Tag it
                    postagged = nltk.pos_tag(tokenized) #Result is a list of
                    #tuples, with word-token and pos-token.
                    logging.debug("\nPOS tagged sentence: " + str(sentence_counter))
                    logging.debug(postagged) #for debug only
                    l = len(postagged)
                    logging.debug("Sentence length: " + str(l))

                    #for i in postagged:
                    for index, token in enumerate(postagged):
                        paper_dict["A_tokens"] += 1 #Increments paper token counter.

                        #THESE LINES SET THE CURRENT AND CONTEXT TOKEN VALUES.
                        this_token = postagged[index]
                        this_type = this_token[0].lower()
                        this_tag = this_token[1]
                        #print("This: " + str(this_type) + "  " + (str(this_tag)))

                        if index > 0:
                            prevToken = postagged[index - 1]
                        else:
                            prevToken = ["NULL","NULL"]
                        prev_type = prevToken[0]
                        prev_tag = prevToken[1]
                        #print("Prev: " + str(prev_type) + "  " + (str(prev_tag)))

                        if index < (l - 1):
                            token1 = postagged[index + 1]
                        else:
                            token1 = ["NULL","NULL"]
                        t1_type = token1[0]
                        t1_tag = token1[1]
                        #print("T1: " + str(t1_type) + "  " + (str(t1_tag)))

                        if index < (l - 2):
                            token2 = postagged[index + 2]
                        else:
                            token2 = ["NULL","NULL"]
                        t2_type = token2[0]
                        t2_tag = token2[1]
                        #print("T2: " + str(t2_type) + "  " + (str(t2_tag)))

                        if index < (l - 3):
                            token3 = postagged[index + 3]
                        else:
                            token3 = ["NULL","NULL"]
                        t3_type = token3[0]
                        t3_tag = token3[1]
                        #print("T3: " + str(t3_type) + "  " + (str(t3_tag)))

                        #print("T4 if")
                        if index < (l - 4):
                            token4 = postagged[index + 4]
                        else:
                            #print("T4 else")
                            token4 = ["NULL","NULL"]
                        #print("T4 assignments")
                        t4_type = token4[0]
                        t4_tag = token4[1]
                        #print("T4: " + str(t4_type) + "  " + (str(t4_tag)))
                        #DONE SETTING TOKEN VALUES.

                        if not (this_type in allpunct):
                            paper_dict["A_words"] += 1
                            aggWordLength += len(this_type)
                        if not (paper_dict["A_words"]>400):
                            if this_type not in first400TTypes:
                                first400TTypes.append(this_type)
                        if this_type in privateVerbs:
                            paper_dict["02PrivateVerbs"] += 1
                        if ((#Condition 1
                                (this_type in priPubSuaVerbs) and
                                ((t1_type in demonstrative) or (t1_type in subjectPro))) or
                            (#Condition 2
                                (this_type in priPubSuaVerbs) and
                                (t1_tag in ["PRP", "NN", "NNS", "NNP", "NNPS"]) and
                                (t2_tag in ["MD", "VB", "VBD", "VBG", "VBN", "VBP"]) ) or
                            (#Condition 3a
                                (this_type in priPubSuaVerbs) and
                                (t1_tag in ["JJ", "JJR", "JJS", "RB", "RBR", "RBS",
                                    "DT", "PRP$"]) and
                                (t2_tag in ["NN", "NNS", "NNP", "NNPS"]) and
                                (t3_tag in ["MD", "VB", "VBD", "VBG", "VBN", "VBP"]) ) or
                            (#Condition 3b
                                (this_type in priPubSuaVerbs) and
                                (t1_tag in ["JJ", "JJR", "JJS", "RB", "RBR", "RBS",
                                    "DT", "PRP$"]) and
                                (t2_tag in ["JJ", "JJR", "JJS"]) and
                                (t3_tag in ["NN", "NNS", "NNP", "NNPS"]) and
                                (t4_tag in ["MD", "VB", "VBD", "VBG", "VBN", "VBP"]) )
                                ):
                            logging.debug("THAT deletion: " + this_type + " " + t1_type
                                    + " " + t2_type + " " + t3_type + " " + t4_type)
                            paper_dict["03ThatDeletion"] += 1
                        if this_type in contracts:
                            paper_dict["04Contractions"] += 1
                        if this_tag in ['VBP', 'VBZ']:
                            paper_dict["05PresVerbs"] += 1
                        if this_type in secPersPro:
                            paper_dict["06SecPersPrn"] += 1
                        if ( this_type in ["do", "does", "did", "doing", "done"] and not
                            ((#Condition 1a
                                (t1_type in ["VB", "VBD", "VBG", "VBN", "VBP"]) ) or
                            (#Condition 1b
                                (t1_type in ["RB", "RBR", "RBS"]) and
                                (t2_type in ["VB", "VBD", "VBG", "VBN", "VBP"]) ) or
                            (#Condition 2
                                (prev_type in allP_whP) or
                                (prev_type == "NULL") )
                                )):
                            paper_dict["07DOproverb"] += 1
                        if this_type in analyticNeg:
                            paper_dict["08AnalyticNeg"] += 1
                        if ((this_type in demonstrative) and
                            ( (t1_tag in ["MD", "VB", "VBD", "VBG", "VBN", "VBP"]) or
                                (t1_type == "and") or
                                (t1_type in wHPWords))):
                            paper_dict["9DemoPron"] += 1
                            logging.debug("Demonstrative pron: " + prev_type + " " +
                                this_type + " " + t1_type + " " + t2_type)
                        if ( (#Condition 1
                                (this_type in ["just", "really", "most", "more"]) ) or
                            (#Condition 2
                                (this_type == "for" and t1_type == "sure") or
                                (this_type == "a" and t1_type == "lot") or
                                (this_type == "such" and t1_type == "a") or
                                (this_type in ["real", "so"] and t1_tag == "JJ") or
                                (this_type in ["do", "does", "did", "doing", "done"]
                                    and t1_type in ["VB", "VBD", "VBG", "VBN", "VBP"])
                                )):
                            paper_dict["10GenEmphatics"] += 1
                            logging.debug("General emphatic: " + this_type + " " + t1_type)
                        if this_type in firstPersPro:
                            paper_dict["11FirstPersPrn"] += 1
                        if this_type == "it":
                            paper_dict["12It"] += 1
                        if (this_type in ["be", "is", "are", "was", "were",
                            "been", "being"] and
                            t1_tag in ["DT", "PRP$", "IN", "JJ", "JJR", "JJS"]):
                            paper_dict["13BeMain"] += 1
                            logging.debug("BE main verb: " + this_type + " " + t1_type)
                        if this_type == "because":
                            paper_dict["14CauseSub"] += 1
                        if (this_type in ["well", "now", "anyway", "anyhow", "anyways"]
                                and (prev_type in clpunct or prev_type == "NULL")):
                            paper_dict["15DiscPart"] += 1
                            logging.debug("Discourse particle: " + prev_type + " " + this_type)
                        if (this_type in indefPro or
                                this_type == "no" and t1_type == "one"):
                            paper_dict["16IndefPro"] += 1
                            logging.debug("Indef pronoun: " + this_type + " " + t1_type)
                        if (this_type in ['almost', 'maybe'] or
                            ((this_type == "at" and t1_type == "about") or
                            (this_type == "something" and t1_type == "like") or
                            (this_type == "more" and t1_type == "or" and t2_type == "less")
                            )
                                ):
                            paper_dict["17GenHedges"] += 1
                            logging.debug("General hedge: " + this_type + " " + t1_type +
                                " " + t2_type)
                        if this_type in amplifiers:
                            paper_dict["18Amplifiers"] += 1
                        if (prev_type == "," and this_type == "which"):
                            paper_dict["19SentRelatives"] += 1
                            logging.debug("Sentence relative: " + prev_type + " " + this_type +
                                " " + t1_type)
                        if (this_type in whoWords and
                                (prev_type == "NULL" or prev_type in clpunct) and
                                (t1_tag == "MD")):
                            paper_dict["20WhQuestion"] += 1
                            logging.debug("WH question: " + prev_type + " " + this_type +
                                " " + t1_type)
                        if this_type in ['can', 'could', 'may', 'might']:
                            paper_dict["21PossModals"] += 1
                        if ( ((this_type in wHPWords) or (this_type in whoWords)) and
                                (prev_type in priPubSuaVerbs) and
                                not (t1_tag == "MD")
                            ):
                            paper_dict["23WhClauses"] += 1
                            logging.debug("WH clause: " + prev_type + " " + this_type +
                                " " + t1_type)
                        if (this_tag == "IN" and (t1_type in clpunct or t1_type == ",")
                                ):
                            paper_dict["24FinalPreps"] += 1
                            logging.debug("Final preposition: " + this_type + " " + t1_type)
                        if this_tag in ['RB', 'RBR', 'RBS']:
                            paper_dict["25Adverbs"] += 1
                        if this_tag in ['NN', 'NNP', 'NNS', 'NNPS']:
                            paper_dict["26Nouns"] += 1
                        if this_tag == 'IN':
                            paper_dict["28Preps"] += 1
                        if (this_tag in ["JJ", "JJR", "JJS"]
                            and t1_tag in ["JJ", "JJR", "JJS", "NN", "NNS",
                                "NNP", "NNPS"]):
                            paper_dict["30AttribAdj"] += 1
                    sentence_counter += 1
                paper_dict["29TTRatio"] =  len(first400TTypes)/4
                paper_dict["27WordLength"] = aggWordLength/paper_dict["A_words"]
                paper_dict["28Preps"] = paper_dict["28Preps"] - paper_dict["24FinalPreps"]

                #Normalize values
                #First ID keys that don't get normalized
                noNormalize = ["A_papernum", "A_tokens", "A_sents", "A_words", "01Gender",
                    "27WordLength", "29TTRatio"]

                normalTokens = paper_dict["A_tokens"]
                #print("Number of tokens: " + str(normalTokens))

                for key, val in paper_dict.items():
                    #print(key + " Value: " + str(val))
                    if key not in noNormalize:
                        #print(key + " needs normalizing")
                        paper_dict[key] = val/normalTokens*1000
                        #print(paper_dict[key])
#                        print(i)


                #Now we need to normalize the data.
                #1. Multiply value of each of the following types of feature by 1/#tokens: functionwords, postags.
                #2. Multiply value of each of trigrams 4/#tokens
                #3. Multiply value of each of bigrams 2/#tokens
                # tokensfactor = 1 / paper_dict["A_tokens"]
                # bigramsfactor = tokensfactor * 2
                # trigramsfactor = tokensfactor * 4
                # print("Factors: Tokens = " + str(tokensfactor) + "  Bigrams = " + str(bigramsfactor) + "  Trigrams = " + str(trigramsfactor))
                # for i in paper_dict.keys():
                #     if i.startswith("F_") or i.startswith("POS_"):
                #         paper_dict[i] = paper_dict[i] * tokensfactor
                #     if i.startswith("Bi_"):
                #         paper_dict[i] = paper_dict[i] * bigramsfactor
                #     if i.startswith("Tri_"):
                #         paper_dict[i] = paper_dict[i] * trigramsfactor
                #
                #Use following three lines for debug only
                #print "Feature set:"
                #for key in sorted(paper_dict.iterkeys()):
                #    print "%s: %s" % (key, paper_dict[key])

                papers.append(paper_dict)

        with open(results_file, "wb") as resPickle:
            pickle.dump(papers, resPickle)

        with open(csv_out_file, "w") as csv_out:
            headers = list(papers[1].keys())
            csvwriter = csv.DictWriter(csv_out, delimiter=',', fieldnames=headers)
            csvwriter.writerow(dict((fn,fn) for fn in headers))
            for row in papers:
                csvwriter.writerow(row)

process("Nonfacttext")
process("Facttext")
process("Fulltext")
