import argparse
import os
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

def load_template(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    return ""

def combine_templates(account, role, user):
    account_template_path = f"Prompt_Templates/accounts/{account}/template.txt"
    role_template_path = f"Prompt_Templates/roles/{role}/template.txt"
    user_template_path = f"Prompt_Templates/users/{user}/template.txt"
    
    account_template = load_template(account_template_path)
    role_template = load_template(role_template_path)
    user_template = load_template(user_template_path)
    
    combined_template = account_template + "\n" + role_template + "\n" + user_template + "\n" + PROMPT_TEMPLATE
    
    return combined_template

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
You are a technical professional looking for the correct commands or process to use based on your role and the account.
Any account restrictions must be applied to appplicable commands
Account instructions take precedence over role instructions. 
Restrictions that apply should be displayed as part of teh response. 
Role commands and processes need to shown exactly do not paraphrase to chnage the meaning
The original question must be part of the prompt pass to the LLM

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    #std_template = "Context : you are a technical professional looking for the coorect commands or process to use based on your role of "+role+" and for account "+account+".  Any account restrictions must be applied to appplicable commands and account instructions take precedence over role instrcutions.  Role commands and processes need to shown exactly as the appear without paraphrasing "

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--role", type=str, required=True, help="Role for the query.")
    parser.add_argument("--user", type=str, required=True, help="User for the query.")
    parser.add_argument("--account", type=str, required=True, help="Account for the query.")
    args = parser.parse_args()

    query_text = args.query_text
    role = args.role
    user = args.user
    account = args.account

    query_rag(query_text, user, role, account)

def build_query(user_query, role, account):
    # Combine role and account with the user query
    #query = f"Role: {role}, Account: {account}, Query: {user_query}"
    query = f"Role: {role}, Account: {account}, Query: {user_query}"
    return query

def query_rag(query_text: str, user: str, role: str, account: str):
    print("Original Query:", query_text)

    # Build the final query with role and account
    final_query = build_query(query_text, role, account)
    print("Final Query:", final_query)

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform the search without a specific role filter
    results = db.similarity_search_with_score(final_query, k=5)
    print("Raw Results Retrieved:", results)

    if not results:
        print("No results found with the current query.")
        return "No valid documents available."

    # Manually filter results by role in the document metadata
    filtered_results = [
        (doc, score) for doc, score in results if (doc.metadata.get("role") == role or doc.metadata.get("account") == account)
    ]
    print("Filtered Results:", filtered_results)

    if not filtered_results:
        print("No results found after filtering by role.")
        return "No valid documents available."

    context_texts = [doc.page_content for doc, _score in filtered_results]
    context_text = "\n\n---\n\n".join(context_texts)
    print("Context Text:", context_text)

    # Combine templates for prompt engineering
    prompt_template = combine_templates(account, role, user)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    final_prompt = prompt.format(context=context_text, question=query_text)
    print("Final Prompt Sent to LLM:", final_prompt)
    
    # Call the model with the constructed prompt
    model = Ollama(model="mistral")
    response_text = model.invoke(final_prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in filtered_results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print("Formatted Response:", formatted_response)
    return formatted_response

if __name__ == "__main__":
    main()
  