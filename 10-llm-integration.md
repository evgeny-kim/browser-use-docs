# 10. LLM Model Integration

Browser-Use is designed to work with a variety of large language models (LLMs) to power its agents. This chapter explores how to integrate different LLM providers and models with Browser-Use.

## Supported LLM Providers

Browser-Use works with multiple LLM providers through the LangChain integration library. Here are the main supported providers:

1. OpenAI (GPT models)
2. Anthropic (Claude models)
3. Google (Gemini models)
4. Azure OpenAI
5. AWS Bedrock
6. Open-source models via Ollama

## OpenAI Models

OpenAI's GPT models, particularly GPT-4o, offer excellent performance for browser automation tasks:

```python
# From examples/models/gpt-4o.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent

# Initialize the OpenAI model
llm = ChatOpenAI(model='gpt-4o')

# Create an agent with the model
agent = Agent(
    task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
    llm=llm,
)

async def main():
    await agent.run(max_steps=10)

asyncio.run(main())
```

Key parameters for OpenAI models:

- `model`: The model name (e.g., 'gpt-4o', 'gpt-3.5-turbo')
- `temperature`: Controls randomness (0.0 to 1.0)
- `max_tokens`: Maximum tokens in the response
- `api_key`: API key (or use environment variable `OPENAI_API_KEY`)

## Claude Models

Anthropic's Claude models offer strong capabilities for browser automation:

```python
# From examples/models/claude-3.7-sonnet.py
import asyncio
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from browser_use import Agent

# Load environment variables
load_dotenv()

# Initialize the Claude model
llm = ChatAnthropic(
    model_name='claude-3-7-sonnet-20250219',
    temperature=0.0,
    timeout=30,
    stop=None
)

# Create an agent with the model
agent = Agent(
    task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
    llm=llm,
)

async def main():
    await agent.run(max_steps=10)

asyncio.run(main())
```

Key parameters for Claude models:

- `model_name`: The Claude model to use (e.g., 'claude-3-7-sonnet-20250219', 'claude-3-opus')
- `temperature`: Controls randomness (0.0 to 1.0)
- `timeout`: Request timeout in seconds
- API key is taken from environment variable `ANTHROPIC_API_KEY`

## Google Gemini Models

Google's Gemini models can be used with Browser-Use for browser automation:

```python
# From examples/models/gemini.py
import asyncio
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig

# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError('GEMINI_API_KEY is not set')

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-exp',
    api_key=SecretStr(api_key)
)

# Configure browser
browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig(
            viewport_expansion=0,
        )
    )
)

async def run_search():
    agent = Agent(
        task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
        llm=llm,
        max_actions_per_step=4,  # Limit actions per step for Gemini
        browser=browser,
    )

    await agent.run(max_steps=25)

asyncio.run(run_search())
```

Key parameters for Gemini models:

- `model`: The Gemini model to use (e.g., 'gemini-2.0-flash-exp', 'gemini-pro')
- `api_key`: The API key as a SecretStr
- Note: Gemini may need adjusted parameters like `max_actions_per_step` for optimal performance

## Azure OpenAI Integration

For organizations using Azure OpenAI services, Browser-Use provides seamless integration:

```python
# From examples/models/azure_openai.py
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from browser_use import Agent

# Load environment variables
load_dotenv()

# Retrieve Azure-specific environment variables
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

if not azure_openai_api_key or not azure_openai_endpoint:
    raise ValueError('AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT is not set')

# Initialize the Azure OpenAI client
llm = AzureChatOpenAI(
    model_name='gpt-4o',
    openai_api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    deployment_name='gpt-4o',  # Use deployment_name for Azure models
    api_version='2024-08-01-preview',  # Explicitly set the API version
)

agent = Agent(
    task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
    llm=llm,
)

async def main():
    await agent.run(max_steps=10)

asyncio.run(main())
```

Key parameters for Azure OpenAI:

- `model_name`: The base model name
- `deployment_name`: Your Azure deployment name
- `openai_api_key`: Azure OpenAI API key
- `azure_endpoint`: Azure endpoint URL
- `api_version`: Azure OpenAI API version

## AWS Bedrock Integration

Browser-Use supports AWS Bedrock for access to multiple foundation models:

```python
# From examples/models/bedrock_claude.py
import asyncio
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from browser_use import Agent

# Load environment variables
load_dotenv()

# Configure AWS credentials
os.environ["AWS_PROFILE"] = "default"  # Or your AWS profile
os.environ["AWS_REGION"] = "us-east-1"  # Your AWS region

# Initialize the Bedrock client for Claude
llm = ChatBedrock(
    model_id="anthropic.claude-3-7-haiku-20240307-v1:0",
    model_kwargs={
        "temperature": 0.0,
        "max_tokens": 4096,
    }
)

agent = Agent(
    task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
    llm=llm,
)

async def main():
    await agent.run(max_steps=10)

asyncio.run(main())
```

Key parameters for AWS Bedrock:

- `model_id`: The Bedrock model ID
- `model_kwargs`: Model-specific parameters
- AWS credentials configured through environment variables or AWS CLI profiles

## Open Source Models via Ollama

Browser-Use can work with open source models through Ollama:

```python
# From examples/models/ollama.py
import asyncio
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from browser_use import Agent

# Load environment variables
load_dotenv()

# Initialize the Ollama client
llm = ChatOllama(
    model="llama3",  # Use any model available in your Ollama installation
    base_url="http://localhost:11434",  # Default Ollama URL
    temperature=0,
)

agent = Agent(
    task='Go to amazon.com, search for laptop, and tell me the name of the first result',
    llm=llm,
)

async def main():
    # Use more steps and actions for local models
    await agent.run(max_steps=20)

asyncio.run(main())
```

Key parameters for Ollama:

- `model`: The name of the model you have pulled in Ollama
- `base_url`: URL where Ollama is running
- `temperature`: Controls randomness
- Note: Open source models may require more steps/actions and simpler tasks

## Custom LLM Integration

You can integrate custom LLMs by implementing the LangChain interface:

```python
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional

class CustomLLM(LLM):
    """Custom LLM implementation."""

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Implement your custom LLM logic here
        # This could call a local API, another service, etc.
        return "Your custom response"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters."""
        return {"model": "custom_model"}

# Use your custom LLM with Browser-Use
custom_llm = CustomLLM()
agent = Agent(
    task="Your task here",
    llm=custom_llm,
)
```

## Model Performance Considerations

Different models have varying strengths and weaknesses for browser automation tasks:

### GPT-4o

- Strengths: Excellent visual understanding, reliable reasoning, good at handling complex websites
- Limitations: API cost, rate limits
- Ideal for: Complex tasks, visual reasoning, production use cases

### Claude 3.7 Sonnet/Opus

- Strengths: Long context window, strong reasoning capabilities, good at following instructions
- Limitations: Slightly less visual understanding than GPT-4o in some cases, API cost
- Ideal for: Tasks with long content, complex reasoning tasks

### Gemini

- Strengths: Good visual reasoning, competitive pricing
- Limitations: May need more steps for complex tasks
- Ideal for: General automation tasks with visual elements

### Open Source Models

- Strengths: Local deployment, no API costs, privacy
- Limitations: Generally less capable than commercial models, require more compute resources
- Ideal for: Simple tasks, development/testing, privacy-sensitive applications

## Model Selection Guidelines

Consider these factors when selecting a model:

1. **Task complexity**: More complex tasks benefit from more capable models like GPT-4o or Claude 3.7
2. **Visual reasoning**: Tasks with complex UIs benefit from models with strong visual capabilities
3. **Cost constraints**: Open source models have no API costs but may require more compute
4. **Privacy requirements**: On-premises models offer better privacy guarantees
5. **Latency needs**: Locally deployed models may offer lower latency

## Optimizing Agent Parameters for Different Models

Adjust agent parameters based on your chosen model:

```python
# For advanced models like GPT-4o
agent = Agent(
    task="Your complex task",
    llm=ChatOpenAI(model="gpt-4o"),
    max_steps=10,  # Standard number of steps is often sufficient
)

# For less capable models
agent = Agent(
    task="Your task, broken down into simpler steps",
    llm=ChatOllama(model="llama3"),
    max_steps=20,  # Allow more steps
    max_actions_per_step=2,  # Limit actions per step
    validate_output=True,  # Add validation to ensure quality
)
```

## Conclusion

Browser-Use's flexible LLM integration allows you to select the model that best fits your specific requirements, balancing factors like capability, cost, and privacy. By understanding each model's strengths and limitations, you can optimize your browser automation tasks for the best results.

In the next chapter, we'll explore UI integrations for Browser-Use, including command line interfaces, Streamlit, and Gradio.
