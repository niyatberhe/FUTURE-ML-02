# Task 2: Automated Customer Support Ticket Classification

## 📌 Project Overview
This project establishes an automated triage pipeline for incoming customer support tickets. Using classic Natural Language Processing (NLP) techniques and traditional linear classification models, the pipeline parses raw text descriptions to concurrently execute two critical support center operations:
1. **Department Routing:** Predicting the `Ticket Type` (e.g., Billing inquiry, Technical issue) to route requests to the correct team.
2. **SLA Prioritization:** Predicting the `Ticket Priority` (e.g., Low, Medium, High, Critical) to manage urgency queues.

---

## 📊 Dataset & Dependency Configuration

### 1. Data Source
The machine learning models are trained on the **Customer Support Ticket Dataset**, which captures historical customer ticket telemetry including metadata, textual logs, and satisfaction metrics.
* **Download Link:** [Kaggle Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)

### 2. Local Repository Setup & `.gitignore`
To adhere to standard source-control best practices and optimize repository size, the raw data file (`customer_support_tickets.csv`) is ignored from version control. 

To run this pipeline locally, place the downloaded CSV file directly into your root directory:
```text
├── task2-model.py
├── customer_support_tickets.csv  
├── .gitignore
└── README.md