import openai
import pandas as pd

# Set up your OpenAI API key
openai.api_key = 'your_openai_api_key'

def query_llm(prompt):
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

def create_prompt(question, data_summary):
    prompt = f"""
    You are a data analysis assistant. Here is a summary of the dataset:
    {data_summary}

    Answer the following question based on the data:
    {question}
    """
    return prompt

def main():
    # Load the preprocessed data
    df = pd.read_csv('processed_data.csv')
    
    # Create a summary of the dataset
    data_summary = df.head().to_string()  # Example summary; adjust as needed
    
    # Ask a detailed question
    question = "What is the target for 'Managed Print Service' contract?"

    # Create prompt and query LLM
    prompt = create_prompt(question, data_summary)
    answer = query_llm(prompt)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
