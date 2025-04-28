# 2. Getting Started with Browser-Use

This chapter will guide you through the process of setting up Browser-Use, configuring your environment, and running your first browser automation task.

## Installation

Browser-Use can be installed using pip, Python's package manager:

```bash
pip install browser-use
```

For the latest development version, you can install directly from the GitHub repository:

```bash
pip install git+https://github.com/browser-use/browser-use.git
```

### Dependencies

Browser-Use requires:

- Python 3.10 or higher
- An LLM provider (OpenAI, Anthropic, etc.)
- Chrome or Chromium browser (Firefox support coming soon)

## Environment Setup

### API Keys

Browser-Use relies on LLMs for its operation. You'll need to set up API keys for your chosen LLM provider. The recommended approach is to use environment variables:

1. Create a `.env` file in your project directory:

```
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic (if using Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key

# Azure OpenAI (if using Azure)
AZURE_OPENAI_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
```

2. Load the environment variables in your script:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Browser Configuration

By default, Browser-Use will use a headless browser. For development and debugging, you may want to see the browser window:

```python
from browser_use import Agent, Browser, BrowserConfig

browser = Browser(
    config=BrowserConfig(
        headless=False,  # Show the browser window
        browser_binary_path='/path/to/chrome',  # Optional: specify Chrome binary
    )
)

agent = Agent(
    task="Your task here",
    llm=your_llm_instance,
    browser=browser
)
```

## Basic Usage

Let's walk through a simple example of using Browser-Use to perform a web search:

```python
import os
import sys
import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0.0,
)

# Define the task
task = 'Search for recent developments in quantum computing and summarize the top 3 findings'

# Create and run the agent
async def main():
    agent = Agent(task=task, llm=llm)
    await agent.run()

if __name__ == '__main__':
    asyncio.run(main())
```

### Understanding the Example

1. **Environment Setup**: We load environment variables containing our API keys.
2. **LLM Configuration**: We initialize a ChatOpenAI instance using GPT-4o.
3. **Task Definition**: We define a natural language task for the agent.
4. **Agent Creation**: We create an Agent instance with our task and LLM.
5. **Execution**: We run the agent asynchronously, which will:
   - Launch a browser
   - Navigate to a search engine
   - Search for quantum computing developments
   - Read and analyze the results
   - Generate a summary

## Agent Parameters

The `Agent` class accepts several parameters to customize its behavior:

```python
agent = Agent(
    task="Your task description",         # The task to perform (required)
    llm=your_llm_instance,                # LLM instance (required)
    browser=browser_instance,             # Optional custom browser instance
    extend_system_message="Custom instructions",  # Add to system prompt
    override_system_message="New prompt", # Replace system prompt
    validate_output=True,                 # Enable output validation
)
```

## Running the Agent

The `run()` method executes the agent's task:

```python
# Run indefinitely or until task completion
result = await agent.run()

# Run with step limit
result = await agent.run(max_steps=10)

# Access the result
print(result.final_answer)  # The agent's final response
```

## Next Steps

Now that you have Browser-Use installed and running, you can explore more advanced features:

- Customizing browser configuration (Chapter 3)
- Implementing custom browser functions (Chapter 6)
- Integrating with different LLM providers (Chapter 10)
- Building UI for your browser agents (Chapter 11)

In the next chapter, we'll delve into the core components of Browser-Use and how they interact to enable browser automation.
