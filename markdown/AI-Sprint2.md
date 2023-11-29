---
marp: true
theme: uncover
class: invert
paginate: true
---
# Goal: Endow AI with our domain knowledge

* domain knowledge source ?

---
# Documents

* meaningless documents: for example, info-help, Confluence websites

* reviewed documents: plain text with an introduction, explanation, and summary structure, for example: Harry Potter, papers, blog posts 

---
# Question-answer data ?

* Manual collection: create a QA website to collect user input data.
 
* Computer collection: find a larger model.
Read reviewed documents and generate QA based on the model's understanding.

---
# What have I done?
* meaningless documents: 
Re-review the meaningless data and try to manually remove garbage data. 

* Use a `MORE-LARGE` open free model understanding to read the slightly cleaned data. Generate QA data.

* Build a train tuning model program
* Tune Model: Failed many times
   - 7B Model: 3300 data: 0.5hr
   - 13B Model: 3300 data: 1hr (33B:2.1hr)
* Help team members

---
# What difficulties did I encounter?

* meaningless documents -> clean documents
* Collecing domain data -> 
  - reviewd and clearly summarized documents source?
  - generate QA data requires manual review 
* Training fine-tuning model
  - Training takes a long time, which is not acceptable to most people? 
  - During training, development and testing cannot be performed.

---
# correct concept

* Model Size: 180B, 130B, 70B, 34B, 13B, 7B
  - \>100B: Models with more than 100B parameters exhibit the phenomenon of emergent intelligence.

* Our hardware limitations:
  - only use \<=34B models: understanding and search and so task
(some text processing)
  - fine-tune \<=13B model