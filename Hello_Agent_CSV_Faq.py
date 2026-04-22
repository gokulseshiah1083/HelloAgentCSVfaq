import pandas as pd
import streamlit as st
import os
import sys
import gdown
from getpass import getpass
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# ==========================================
#  PART 1: AUTOMATIC FILE DOWNLOADER
# ==========================================

streamlit_title = "Hello Agent: Multi-File CSV FAQ"
st.title(streamlit_title)
st.write("This app demonstrates how to create an AI agent that can read and answer questions from multiple CSV files. The agent will determine which file is relevant to the user's question and provide an answer based on the data in that file.")

files_to_download = {
    "saas_docs.csv":         "https://drive.google.com/file/d/1RElOhN7bYsDAJUNQhYyqM7IzX-Xo6myq/view?usp=sharing",
    "credit_card_terms.csv": "https://drive.google.com/file/d/1_giivc_B0urOKpct0XY2yVZuxW3Eenuf/view?usp=sharing",
    "hospital_policy.csv":   "https://drive.google.com/file/d/1pL7OnDhnmz9pteIpBJ12gu2_ixrc2hPm/view?usp=sharing",
    "ecommerce_faqs.csv":    "https://drive.google.com/file/d/1O4fTjsLFbz55oOiwJUwLwZryO5OSSF6p/view?usp=sharing"
}

print("--- Downloading Files from Google Drive ---")
for filename, url in files_to_download.items():
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)
        print(f"Downloaded: {filename}")
    else:
        print(f"Skipped: {filename} (Already exists)")
print("--- Download Complete ---\n")

# ==========================================
#  PART 2: AI AGENT SETUP (MULTI-FILE)
# ==========================================

# 1. SETUP: Get API Key Securely
print("ENTER YOUR OPENAI API KEY BELOW:")
api_key = getpass()

# 2. LOAD ALL CSVs INTO A LIST
dataframes = [] # We will store all the loaded tables here
loaded_names = []

try:
    for filename in files_to_download.keys():
        df = pd.read_csv(filename)
        dataframes.append(df)
        loaded_names.append(filename)
        print(f"SUCCESS: Loaded '{filename}' ({len(df)} rows)")

except Exception as e:
    print(f"\nERROR loading files: {e}")
    sys.exit()

# 3. DEFINE THE RULES
system_prompt = """
You are a smart data assistant capable of reading multiple CSV files.
- You have access to 4 different datasets: SaaS Docs, Credit Card Terms, Hospital Policy, and Ecommerce FAQs.
- When asked a question, determine which DataFrame is most relevant.
- Do NOT answer from general knowledge.
- Answer in plain English.
"""

try:
    # ---------------------------------------------------------
    # TODO 1: Initialize the LLM
    # Hint: Use ChatOpenAI with model="gpt-4o-mini" and temperature=0.0
    # ---------------------------------------------------------
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key) # <--- REPLACE 'None' WITH YOUR CODE

    # ---------------------------------------------------------
    # TODO 2: Create the Pandas Agent
    # Hint: Pass the 'llm' and the list 'dataframes' as arguments.
    # Set verbose=True and allow_dangerous_code=True
    # ---------------------------------------------------------
    agent = create_pandas_dataframe_agent(
        llm,
        dataframes,  # <--- PASS THE DATAFRAMES LIST HERE
        verbose=True,
        agent_type="openai-functions",
        allow_dangerous_code=True
    )

    print("\nAI Agent is ready! You can ask questions across ALL files.")
    print("Example: 'What is the visiting hour in the hospital?' or 'What is the API limit?'")

except Exception as e:
    print(f"Error initializing agent: {e}")
    sys.exit()

# ==========================================
#  PART 3: CHAT LOOP
# ==========================================
print("\nType 'exit' or 'quit' to stop conversation.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        st.stop() 
        break

    if not user_input:
        continue

    final_query = system_prompt + "\n\nQuestion: " + user_input
    print("AI is thinking...")

    try:
        # ---------------------------------------------------------
        # TODO 4: Invoke the Agent
        # Hint: Use agent.invoke() and pass the final_query
        # The result will be a dictionary, access ['output']
        # ---------------------------------------------------------
        response = agent.invoke(final_query)['output'] # <--- REPLACE THIS WITH YOUR CODE
        
        print(f"AI: {response}\n" + "-"*30)
    except Exception as e:
        print(f"An error occurred: {e}")

