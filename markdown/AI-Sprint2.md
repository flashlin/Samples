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
* Unstructured documents: 
Re-review the Unstructured data and try to manually remove garbage data. 

* Use a `MORE-LARGE` model understanding to read the slightly cleaned data. Generate QA data.

* Build a training model program

---
# What difficulties did I encounter?

* Once offline, Windows 11 seems to pause operations.
* Unstructured documents -> structured documents
* Training fine-tuning model
  - Training takes a long time, which is not acceptable to most people.
  - During training, development and testing cannot be performed.

---
# correct concept
