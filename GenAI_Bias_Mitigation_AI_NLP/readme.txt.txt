This is a replication procedure for the project, by Richard Huang and Jatin Chadha.

1. Install the requirements in 'code\requirements.txt' to a python environment
2. Acquire an official API key from the three LLM sources from the links below:

https://replicate.com/
https://gemini.google.com/
https://openai.com/api/

Our API keys are obfuscated in the notebooks for obvious reasons and you will need to input your own. It is recommended to use a paid/billed account to overcome some of the rate limits.

3. Pick and choose which data you would like to use/test. The folder 'code\winobias_text_data' has all ~3000 sentences of the original Winobias data. To replicate our results, please use the type 2
sentences as indicated. In 'datatables' folder, there are the sentences and winobias dataset csv files.

4. Run inference on the queries. For ease of use, the 'wb_extract' csv file in the 'datatables' folder contains *all* of our work, including the queries used and the results from the queries, sentences, details,
etc.

Note: You may need to comb through the answers to obtain what the LLM predicts if it does not follow the prompting guide perfectly. LLAMA is the worst offender of this.