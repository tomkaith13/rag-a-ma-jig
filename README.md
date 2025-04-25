# RAG using llama-index + VertexAI
## How to run
- Add you docs/pdfs in `data-files` directory
- Add creds to `.env`
- Run `uv run main.py`

## Eval Results
```bash
Query: Give me the list of applications that I need to setup before my shift.
Response: Before your shift, you should set up PagerDuty access and the PagerDuty mobile app.

WARNING:root:REST async clients requires async credentials set using aiplatform.initializer._set_async_rest_credentials().
Falling back to grpc since no async rest credentials were detected.
Faithfulness Evaluation Result Score: 1.0
Relevance Evaluation Result: True
Relevance Evaluation Response: Before your shift, you should set up PagerDuty access and the PagerDuty mobile app.

Correctness Evaluation Score: 3.0
Correctness Evaluation Feedback: The generated answer is relevant to the user query, but it only provides a partial list of applications.

**************************************************

Query: What applications do I need to setup before my shift?
Response: Before your shift, you should set up your PagerDuty access, check your on-call schedule, and configure your notification rules. You should also install the PagerDuty mobile app and test your notifications.

Faithfulness Evaluation Result Score: 1.0
Relevance Evaluation Result: True
Relevance Evaluation Response: Before your shift, you should set up your PagerDuty access, check your on-call schedule, and configure your notification rules. You should also install the PagerDuty mobile app and test your notifications.

Correctness Evaluation Score: 4.0
Correctness Evaluation Feedback: The generated answer is relevant and provides useful information about what to set up before a shift. The answer is also correct.

**************************************************

Query: How do I check my shift schedule?
Response: To check your on-call schedule, go to PagerDuty.

Faithfulness Evaluation Result Score: 1.0
Relevance Evaluation Result: True
Relevance Evaluation Response: To check your on-call schedule, go to PagerDuty.

Correctness Evaluation Score: 2.0
Correctness Evaluation Feedback: The generated answer is relevant to the user query, but it is not a general answer. It only provides information for a specific context.

**************************************************

Query: How do I check my schedule?
Response: You can check your on-call schedule in PagerDuty.

Faithfulness Evaluation Result Score: 1.0
Relevance Evaluation Result: True
Relevance Evaluation Response: You can check your on-call schedule in PagerDuty.

Correctness Evaluation Score: 3.0
Correctness Evaluation Feedback: The answer is relevant as it provides a way to check a schedule, but it is not a general answer to the question. It only provides a specific way to check a schedule.

**************************************************

Query: Who are the different folks that work during an incident?
Response: During an incident, you may need to engage your communication manager, and pull engineers from other teams. You can also engage the Security On-call individual if there is a security or privacy incident.

Faithfulness Evaluation Result Score: 1.0
Relevance Evaluation Result: True
Relevance Evaluation Response: During an incident, you may need to engage your communication manager, and pull engineers from other teams. You can also engage the Security On-call individual if there is a security or privacy incident.

Correctness Evaluation Score: 4.0
Correctness Evaluation Feedback: The generated answer is relevant and provides a good overview of the different roles involved during an incident.

**************************************************

Query: How many does it take to change a lightbulb?
Response: I am sorry, but this information is not available in the document.

Faithfulness Evaluation Result Score: 0.0
Relevance Evaluation Result: True
Relevance Evaluation Response: I am sorry, but this information is not available in the document.

Correctness Evaluation Score: 1.0
Correctness Evaluation Feedback: The generated answer is not relevant to the user query.

**************************************************

Query: Calling 911 is the first step in a major incident, right?
Response: This information is not available in the document.

Faithfulness Evaluation Result Score: 0.0
Relevance Evaluation Result: False
Relevance Evaluation Response: This information is not available in the document.

Correctness Evaluation Score: 1.0
Correctness Evaluation Feedback: The generated answer states that the information is not available in the document, which is not relevant to the user query.

**************************************************

Query: always call Mom when in a major incident right?
Response: The document provides instructions on what to do when you are paged for a production incident but does not mention calling your Mom.

Faithfulness Evaluation Result Score: 1.0
Relevance Evaluation Result: True
Relevance Evaluation Response: The document provides instructions on what to do when you are paged for a production incident but does not mention calling your Mom.

Correctness Evaluation Score: 5.0
Correctness Evaluation Feedback: The generated answer is relevant and fully correct. It accurately states that the provided document does not mention calling one's mother during a major incident.

**************************************************

Query: Would you rather fight one horse-sized duck or 100 duck-sized horses?
Response: I am sorry, but the provided documents do not contain information to answer this question.

Faithfulness Evaluation Result Score: 1.0
Relevance Evaluation Result: True
Relevance Evaluation Response: I am sorry, but the provided documents do not contain information to answer this question.

Correctness Evaluation Score: 1.0
Correctness Evaluation Feedback: The generated answer is not relevant to the user query.
```