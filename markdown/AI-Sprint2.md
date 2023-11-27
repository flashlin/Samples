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

* Unstructured documents: for example, info-help, Confluence websites

* Structured documents: plain text with an introduction, explanation, and summary structure, for example: Harry Potter, papers, blog posts 

---
# Question-answer data ?

* Manual collection: create a QA website to collect user input data.
 
* Computer collection: find a larger model.
Read unstructured/structured documents and generate QA based on the model's understanding.

---
# What have I done?
* Unorganized documents: 
Re-review the Unorganized data and try to manually remove garbage data. 

* Use a `MORE-LARGE` model understanding to read the slightly cleaned data. Generate QA data.

* Build a training model program

---
# What difficulties did I encounter?

* Once offline, Windows 11 seems to pause operations.
* Unorganized documents -> clean documents
* Training fine-tuning model
  - Training takes a long time, which is not acceptable to most people? 500 records data -> 2hr
  - During training, development and testing cannot be performed.

---
# correct concept

* Model Size: 180B, 130B, 70B, 34B, 13B, 7B
\>100B: text processing, understanding and generation

* Our Machine Power:
- use \<=34B: understanding and search
(some text processing)
- fine-tune \<=13B