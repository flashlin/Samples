---
marp: true
theme: uncover
class: invert
paginate: true
---
# Goal: Endow AI with our domain knowledge


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
* Use a `MORE-POWER` Open Model to read the slightly cleaned data. Generate QA data.
* Build a train tuning model program
* Tune Model: Failed many times
   - 7B Model: 3000 data 2hr (13B:4hr, 34B:8hr)

---
# What difficulties
* meaningless documents -> clean documents
* Collecing domain data -> 
  - reviewd and clearly summarized documents source?
  - generate QA data requires manual review 
* Training fine-tuning model
  - Training takes a long time, which is not acceptable to most people? 
  - During training, development and testing cannot be performed.

---
# Conclusion
* Model Size: 180B, 130B, 70B, 34B, 13B, 7B
  - \>100B: Models with more than 100B exhibit the phenomenon of emergent intelligence.
  - Mistral-7B
  - Yi-34B 

* Our hardware limitations:
  - use <=34B Models
