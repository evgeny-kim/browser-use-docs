# 11. UI Integration

Browser-Use can be integrated into various user interfaces to create interactive tools for browser automation. This chapter explores how to incorporate Browser-Use into command-line interfaces, Gradio applications, and Streamlit applications.

## Command Line Interface

Browser-Use can be used to create powerful command-line tools for browser automation. The following example shows how to build a basic command-line interface:

```python
# From examples/ui/command_line.py
import argparse
import asyncio
import os
from dotenv import load_dotenv
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller

load_dotenv()

def get_llm(provider: str):
    if provider == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError('Error: ANTHROPIC_API_KEY is not set. Please provide a valid API key.')
        return ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.0)
    elif provider == 'openai':
        from langchain_openai import ChatOpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('Error: OPENAI_API_KEY is not set. Please provide a valid API key.')
        return ChatOpenAI(model='gpt-4o', temperature=0.0)
    else:
        raise ValueError(f'Unsupported provider: {provider}')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Automate browser tasks using an LLM agent.')
    parser.add_argument(
        '--query', type=str, help='The query to process',
        default='go to reddit and search for posts about browser-use'
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'anthropic'],
        default='openai',
        help='The model provider to use (default: openai)',
    )
    return parser.parse_args()

def initialize_agent(query: str, provider: str):
    """Initialize the browser agent with the given query and provider."""
    llm = get_llm(provider)
    controller = Controller()
    browser = Browser(config=BrowserConfig())

    return Agent(
        task=query,
        llm=llm,
        controller=controller,
        browser=browser,
        use_vision=True,
        max_actions_per_step=1,
    ), browser

async def main():
    """Main async function to run the agent."""
    args = parse_arguments()
    agent, browser = initialize_agent(args.query, args.provider)

    await agent.run(max_steps=25)

    input('Press Enter to close the browser...')
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
```

### Using the Command Line Interface

To use the command-line interface:

```bash
# Example 1: Using OpenAI (default), with default task
python command_line.py

# Example 2: Using OpenAI with a custom query
python command_line.py --query "go to google and search for browser-use"

# Example 3: Using Anthropic's Claude model with a custom query
python command_line.py --query "find latest Python tutorials on Medium" --provider anthropic
```

### Command Line Interface Design Patterns

The example demonstrates several key patterns for command-line interfaces:

1. **Argument parsing** with `argparse` for flexible command-line options
2. **Model provider selection** with the `--provider` argument
3. **Custom query input** with the `--query` argument
4. **Proper cleanup** with browser closing after execution

## Gradio Integration

[Gradio](https://www.gradio.app/) is a popular Python library for creating web interfaces for machine learning models. Browser-Use can be integrated with Gradio to create interactive web applications for browser automation:

```python
# From examples/ui/gradio_demo.py
import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional

import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from browser_use import Agent

load_dotenv()

@dataclass
class ActionResult:
    is_done: bool
    extracted_content: Optional[str]
    error: Optional[str]
    include_in_memory: bool

@dataclass
class AgentHistoryList:
    all_results: List[ActionResult]
    all_model_outputs: List[dict]

def parse_agent_history(history_str: str) -> None:
    console = Console()
    # Split the content into sections based on ActionResult entries
    sections = history_str.split('ActionResult(')

    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        # Extract relevant information
        content = ''
        if 'extracted_content=' in section:
            content = section.split('extracted_content=')[1].split(',')[0].strip("'")

        if content:
            header = Text(f'Step {i}', style='bold blue')
            panel = Panel(content, title=header, border_style='blue')
            console.print(panel)
            console.print()

async def run_browser_task(
    task: str,
    api_key: str,
    model: str = 'gpt-4o',
    headless: bool = True,
) -> str:
    if not api_key.strip():
        return 'Please provide an API key'

    os.environ['OPENAI_API_KEY'] = api_key

    try:
        agent = Agent(
            task=task,
            llm=ChatOpenAI(model='gpt-4o'),
        )
        result = await agent.run()
        return result
    except Exception as e:
        return f'Error: {str(e)}'

def create_ui():
    with gr.Blocks(title='Browser Use GUI') as interface:
        gr.Markdown('# Browser Use Task Automation')

        with gr.Row():
            with gr.Column():
                api_key = gr.Textbox(label='OpenAI API Key', placeholder='sk-...', type='password')
                task = gr.Textbox(
                    label='Task Description',
                    placeholder='E.g., Find flights from New York to London for next week',
                    lines=3,
                )
                model = gr.Dropdown(choices=['gpt-4', 'gpt-3.5-turbo'], label='Model', value='gpt-4')
                headless = gr.Checkbox(label='Run Headless', value=True)
                submit_btn = gr.Button('Run Task')

            with gr.Column():
                output = gr.Textbox(label='Output', lines=10, interactive=False)

        submit_btn.click(
            fn=lambda *args: asyncio.run(run_browser_task(*args)),
            inputs=[task, api_key, model, headless],
            outputs=output,
        )

    return interface

if __name__ == '__main__':
    demo = create_ui()
    demo.launch()
```

### Launching the Gradio Interface

To launch the Gradio interface:

```bash
python gradio_demo.py
```

This will start a local web server, and the interface will be accessible at http://localhost:7860 by default.

### Gradio Interface Features

The example Gradio interface includes:

1. **Input form** with fields for:
   - OpenAI API key
   - Task description
   - Model selection
   - Headless mode toggle
2. **Output display** showing the result of the browser task
3. **Async execution** of the browser task using `asyncio.run()`

## Streamlit Integration

[Streamlit](https://streamlit.io/) is another popular framework for creating web applications. Browser-Use can be integrated with Streamlit for interactive browser automation apps:

```python
# From examples/ui/streamlit_demo.py
import asyncio
import os
import sys

import streamlit as st
from dotenv import load_dotenv

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller

# Load environment variables
load_dotenv()

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Function to get the LLM based on provider
def get_llm(provider: str):
    if provider == 'anthropic':
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            st.error('Error: ANTHROPIC_API_KEY is not set. Please provide a valid API key.')
            st.stop()

        return ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.0)
    elif provider == 'openai':
        from langchain_openai import ChatOpenAI

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error('Error: OPENAI_API_KEY is not set. Please provide a valid API key.')
            st.stop()

        return ChatOpenAI(model='gpt-4o', temperature=0.0)
    else:
        st.error(f'Unsupported provider: {provider}')
        st.stop()

# Function to initialize the agent
def initialize_agent(query: str, provider: str):
    llm = get_llm(provider)
    controller = Controller()
    browser = Browser(config=BrowserConfig())

    return Agent(
        task=query,
        llm=llm,
        controller=controller,
        browser=browser,
        use_vision=True,
        max_actions_per_step=1,
    ), browser

# Streamlit UI
st.title('Automated Browser Agent with LLMs 🤖')

query = st.text_input('Enter your query:', 'go to reddit and search for posts about browser-use')
provider = st.radio('Select LLM Provider:', ['openai', 'anthropic'], index=0)

if st.button('Run Agent'):
    st.write('Initializing agent...')
    agent, browser = initialize_agent(query, provider)

    async def run_agent():
        with st.spinner('Running automation...'):
            await agent.run(max_steps=25)
        st.success('Task completed! 🎉')

    asyncio.run(run_agent())

    st.button('Close Browser', on_click=lambda: asyncio.run(browser.close()))
```

### Running the Streamlit App

To run the Streamlit application:

```bash
python -m streamlit run streamlit_demo.py
```

This will launch a local Streamlit server and open the application in your default web browser.

### Streamlit Interface Features

The Streamlit interface includes:

1. **Input controls**:
   - Text input for the query
   - Radio buttons to select the LLM provider
2. **Interactive components**:
   - Run button to start the agent
   - Close Browser button to clean up resources
3. **Status indicators**:
   - Initializing message
   - Spinner during execution
   - Success message upon completion

## Custom UI Integration

Browser-Use can be integrated into custom UIs by following these general patterns:

1. **Initialize the agent** with the desired configuration
2. **Run the agent asynchronously** to avoid blocking the UI
3. **Handle cleanup** of browser resources
4. **Display results** in an appropriate format

### Example: Basic Integration Pattern

```python
import asyncio
from browser_use import Agent
from your_custom_ui import UI

# Initialize your custom UI
ui = UI()

# Function to run the agent based on user input
async def run_browser_agent(task, model_type):
    # Initialize the appropriate LLM based on user selection
    llm = get_llm_for_model_type(model_type)

    # Create and run the agent
    agent = Agent(task=task, llm=llm)
    result = await agent.run()

    # Update the UI with the result
    ui.display_result(result)

    # Clean up resources
    await agent.close()

# Connect to UI events
ui.on_submit(lambda task, model_type: asyncio.create_task(run_browser_agent(task, model_type)))

# Start the UI
ui.run()
```

## Best Practices for UI Integration

1. **Asynchronous Execution**: Always run Browser-Use agents asynchronously to avoid blocking the UI
2. **Resource Cleanup**: Ensure browsers are properly closed after use
3. **Error Handling**: Provide clear error messages to users when tasks fail
4. **Progress Indication**: Show the status of long-running tasks
5. **Secure API Key Handling**: Use secure methods for handling API keys
6. **User Guidance**: Provide examples and guidance for effective task descriptions

## Conclusion

Browser-Use can be integrated into various UI frameworks to create powerful and interactive browser automation applications. Whether using command-line interfaces, Gradio, Streamlit, or custom UIs, the core patterns of agent initialization, asynchronous execution, and proper resource management remain consistent.

In the next chapter, we'll explore how Browser-Use can be integrated with other services, such as Discord, Slack, and file management systems.
