# 13. Jupyter Notebook Integration

Browser-Use can be seamlessly integrated into Jupyter notebooks, enabling interactive development, experimentation, and demonstration of browser automation tasks. This chapter explores how to use Browser-Use within Jupyter notebooks for efficient workflow development.

## Setting Up Browser-Use in Jupyter

To use Browser-Use in a Jupyter notebook, you need to install the required packages and configure the environment correctly:

```python
# Install required packages
%pip install -U langgraph langchain_google_genai langchain_community langgraph-checkpoint-postgres openai langchain_groq
%pip install --upgrade --quiet playwright lxml browser-use langchain_openai

# Install Playwright browsers
!playwright install

# For specific compatibility issues
!pip install "anyio<4"
```

### Handling Jupyter's Event Loop

Jupyter notebooks use their own event loop, which can conflict with the asynchronous nature of Browser-Use. To resolve this, use `nest_asyncio`:

```python
# This import is required only for jupyter notebooks, since they have their own eventloop
import nest_asyncio
nest_asyncio.apply()
```

## Basic Browser-Use in Jupyter

Here's a simple example of using Browser-Use in a Jupyter notebook:

```python
# From examples/notebook/agent_browsing.ipynb
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig

# Initialize the LLM
llm = ChatOpenAI(model='gpt-4o', temperature=0)

# Basic configuration for the browser
config = BrowserConfig(
    headless=True,  # Run in headless mode
    # disable_security=True  # Uncomment if you want to disable security
)

# Initialize the browser with the specified configuration
browser = Browser(config=config)

async def main():
    # Initialize the agent with the task and language model
    agent = Agent(
        task='What is Langgraph',
        llm=llm,
        browser=browser,
        generate_gif=False,  # Disable GIF generation
    )

    # Run the agent and get results asynchronously
    result = await agent.run()

    # Process results token-wise
    for action in result.action_results():
        print(action.extracted_content, end='\r', flush=True)
        print('\n\n')

    # Close the browser after completion
    await browser.close()

# Run the asynchronous main function
asyncio.run(main())
```

## Executing Single Tasks

For quick experimentation, you can execute single browser automation tasks:

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig

# Configure browser
browser = Browser(config=BrowserConfig(headless=True))

# Create agent
agent = Agent(
    task='what is langchain',
    llm=ChatOpenAI(model='gpt-4o', temperature=0),
    browser=browser,
    generate_gif=False,
)

# Run agent
result = await agent.run()

# Display result
for action in result.action_results():
    if action.is_done:
        print(action.extracted_content)

# Close the browser
await browser.close()
```

## Accessing and Processing Results

Browser-Use agents produce detailed results that can be accessed and processed in various ways:

```python
# Display full history object
print(result)

# Access only completed actions
for action in result.action_results():
    if action.is_done:
        print(action.extracted_content)

# Access all model outputs
for output in result.all_model_outputs:
    print(output)
```

The result object contains:

- `action_results()`: List of action results with extracted content
- `is_done()`: Boolean indicating if the task was completed
- `history`: Complete history of the agent's execution
- `all_model_outputs`: All outputs from the language model

## Visual Feedback

While running Browser-Use in Jupyter, you can get visual feedback on the agent's progress:

```python
# Enable visual feedback with step-by-step output
agent = Agent(
    task='Go to GitHub and find the top trending Python repositories',
    llm=llm,
    browser=browser,
    generate_gif=True,  # Generate a GIF of the browsing session
)

result = await agent.run()

# If GIF generation is enabled, display the GIF
from IPython.display import Image
Image(filename='browsing_session.gif')
```

## Reusing the Browser Instance

You can reuse the same browser instance for multiple tasks to improve efficiency:

```python
# Initialize a browser instance
browser = Browser(config=BrowserConfig(headless=True))

# First task
agent1 = Agent(
    task='Search for the weather in New York',
    llm=llm,
    browser=browser,
)
result1 = await agent1.run()

# Second task using the same browser instance
agent2 = Agent(
    task='Search for the top tourist attractions in New York',
    llm=llm,
    browser=browser,
)
result2 = await agent2.run()

# Close the browser when done
await browser.close()
```

## Interactive Experimentation

Jupyter notebooks are ideal for interactive experimentation with Browser-Use:

```python
# Define a function to run tasks interactively
async def run_browser_task(task):
    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
    )
    result = await agent.run()
    return result

# Create interactive controls for task input
from ipywidgets import widgets
from IPython.display import display

task_input = widgets.Text(
    value='',
    placeholder='Enter a browser task',
    description='Task:',
    disabled=False
)

run_button = widgets.Button(description="Run Task")
output = widgets.Output()

def on_button_click(b):
    with output:
        output.clear_output()
        print("Running task...")
        result = asyncio.run(run_browser_task(task_input.value))
        print("Task completed!")
        for action in result.action_results():
            if action.is_done:
                print(action.extracted_content)

run_button.on_click(on_button_click)
display(task_input, run_button, output)
```

## Best Practices for Jupyter Integration

1. **Resource Management**:

   - Always close browser instances when done to free up resources
   - Use headless mode for faster execution and less resource usage

2. **Error Handling**:

   - Implement try/except blocks to handle errors gracefully
   - Display meaningful error messages to aid debugging

   ```python
   try:
       result = await agent.run()
       # Process result
   except Exception as e:
       print(f"Error occurred: {str(e)}")
   finally:
       await browser.close()
   ```

3. **Visual Output**:

   - Use Jupyter's rich display capabilities for visualizing results
   - Consider using progress bars for long-running tasks

   ```python
   from tqdm.notebook import tqdm

   # Create a progress indicator
   with tqdm(total=1) as pbar:
       result = await agent.run()
       pbar.update(1)
   ```

4. **Session Management**:

   - Store and retrieve session data for continuity between notebook runs
   - Save important results to variables or files

5. **Performance Optimization**:
   - Use `max_steps` parameter to limit execution time
   - Consider running resource-intensive operations in a separate process

## Example Notebook Workflow

Here's a complete example of a typical Browser-Use workflow in a Jupyter notebook:

```python
import asyncio
import nest_asyncio
from IPython.display import display, HTML
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

# Configure and initialize browser
browser_config = BrowserConfig(
    headless=True,
    disable_security=False,
)
browser = Browser(config=browser_config)

# Initialize LLM
llm = ChatOpenAI(model='gpt-4o', temperature=0)

# Define tasks
tasks = [
    "Search for 'Browser automation best practices' and summarize the top results",
    "Go to GitHub and find the most starred Python repositories",
    "Go to Weather.com and get the weather forecast for New York City"
]

# Run tasks sequentially
async def run_tasks():
    results = []
    for i, task in enumerate(tasks):
        print(f"Running task {i+1}/{len(tasks)}: {task}")

        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
        )

        try:
            result = await agent.run(max_steps=15)
            results.append(result)

            # Display result
            for action in result.action_results():
                if action.is_done:
                    display(HTML(f"<b>Task {i+1} Result:</b> {action.extracted_content}"))
        except Exception as e:
            print(f"Error in task {i+1}: {str(e)}")

    return results

# Execute all tasks
all_results = asyncio.run(run_tasks())

# Close browser when done
asyncio.run(browser.close())

# Save results for further analysis
import json
with open('browsing_results.json', 'w') as f:
    json.dump([str(r) for r in all_results], f)
```

## Conclusion

Jupyter notebooks provide an ideal environment for developing, testing, and demonstrating Browser-Use automation workflows. By leveraging Jupyter's interactive nature, you can iteratively refine your browser automation tasks and visualize the results in real-time. The combination of Browser-Use's powerful automation capabilities with Jupyter's interactive development environment creates a flexible and efficient platform for web automation projects.

In the next chapter, we'll explore performance and optimization techniques for Browser-Use applications.
