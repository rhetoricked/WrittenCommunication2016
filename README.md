# GenderGenre Written Communication 2016
This is an update to Python 3.5 of the code originally used for analyzing
writing samples of first-year law students and to generate the data for the article
Gender/Genre: The Lack of Gendered Register in Texts Requiring Genre Knowledge.
_Written Communication_, 33(4), 360–384. https://doi.org/10.1177/0741088316667927
(the "Article"). If you use this code, you should cite to the Article.

The original code used for earlier work appears in the repository titled
GenderGenreOriginal at https://github.com/rhetoricked/GenderGenreOriginal That
repository is dated and performs different analyses than those used for the
Article. This repository has all the code for the Article.

Data collection and preparation, including human coding of key text spans, are
described in the Article and in Larson, B. (2015, May). Gender/Genre: Gender
difference in disciplinary communication (Ph.D. dissertation). University of
Minnesota, Minneapolis. (Dissertation subject to embargo until May 2017.)

The original text artifacts for this study were Microsoft Word files (and one
PDF converted by the researcher to Word). They were manually annotated and
converted into XML files using GATE natural language processing software.
The original and XML files will be available from the Linguistic Data
Consortium in spring 2017. If you need them before then or cannot access LDC
for whatever reason, contact Brian Larson.

To use this code, you need directories set up with the structure indicated
in the code modules, and you need to have the XML data files resident in the
appropriate spot, again indicated in the code modules. You then run modules
in the following order:
Module1.py
Module2.py
Module3Biber.py
Module4BiberStats.py
