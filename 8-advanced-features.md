# 8. Advanced Features

Browser-Use includes numerous advanced features that extend its capabilities beyond basic browser automation. This chapter explores these advanced features and how they can enhance your automated browsing experience.

## Multi-Tab Handling

Browser-Use supports working with multiple tabs, allowing agents to navigate between different pages while maintaining context:

```python
# From examples/features/multi-tab_handling.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent

# Create an agent that will work with multiple tabs
llm = ChatOpenAI(model='gpt-4o')
agent = Agent(
    task='open 3 tabs with elon musk, trump, and steve jobs, then go back to the first and stop',
    llm=llm,
)

async def main():
    await agent.run()

asyncio.run(main())
```

Multi-tab functionality includes:

- Opening new tabs
- Navigating between existing tabs
- Closing specific tabs
- Working with content across multiple tabs
- Maintaining tab state during navigation

The LLM agent understands tab context and can perform operations like:

```
"Open a new tab and go to wikipedia.org"
"Go back to the first tab"
"Close the current tab"
"Compare information between tab 1 and tab 3"
```

## Cross-Origin iFrames Support

Browser-Use can interact with iFrames, including those from different origins:

```python
# From examples/features/cross_origin_iframes.py
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig, Controller

# Create a browser instance
browser = Browser(
    config=BrowserConfig(
        browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    )
)
controller = Controller()

async def main():
    # Create an agent that will work with iframes
    agent = Agent(
        task='Click "Go cross-site (simple page)" button on https://csreis.github.io/tests/cross-site-iframe.html then tell me the text within',
        llm=ChatOpenAI(model='gpt-4o', temperature=0.0),
        controller=controller,
        browser=browser,
    )

    await agent.run()
    await browser.close()

asyncio.run(main())
```

Key iframe capabilities:

- Detecting and navigating iframes
- Interacting with elements inside iframes
- Handling cross-origin restrictions
- Extracting content from nested iframes

## Custom System Prompts

Browser-Use allows you to customize the system prompt that guides the agent's behavior:

```python
# From examples/features/custom_system_prompt.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent

# Define a custom extension to the system prompt
extend_system_message = (
    'REMEMBER the most important RULE: ALWAYS open first a new tab and go first to url wikipedia.com no matter the task!!!'
)

async def main():
    task = "do google search to find images of Elon Musk's wife"
    model = ChatOpenAI(model='gpt-4o')

    # Add the custom instructions to the system prompt
    agent = Agent(
        task=task,
        llm=model,
        extend_system_message=extend_system_message
    )

    await agent.run()

asyncio.run(main())
```

There are two ways to customize the system prompt:

1. `extend_system_message`: Append additional instructions to the default prompt
2. `override_system_message`: Replace the entire system prompt with your own

Use cases for custom system prompts:

- Adding specific behavioral rules
- Setting preferences for website navigation
- Providing domain-specific knowledge
- Enforcing safety constraints
- Establishing interaction patterns

## Element Interaction Fallbacks

Browser-Use implements robust fallback mechanisms for interacting with web elements:

```python
# From examples/features/click_fallback_options.py
async def main():
    # Different ways to target elements
    xpath_task = 'Open http://localhost:8000/, click element with the xpath "/html/body/div/div[1]" and then click on Oranges'
    css_selector_task = 'Open http://localhost:8000/, click element with the selector div.select-display and then click on apples'
    text_task = 'Open http://localhost:8000/, click the third element with the text "Select a fruit" and then click on Apples'
    select_task = 'Open http://localhost:8000/, choose the car BMW'
    button_task = 'Open http://localhost:8000/, click on the button'

    # Run an agent for each task
    for task in [xpath_task, css_selector_task, text_task, select_task, button_task]:
        agent = Agent(task=task, llm=llm)
        await agent.run()
```

Fallback interaction methods include:

1. Standard click on visible elements
2. JavaScript click execution
3. XPath-based interactions
4. CSS selector targeting
5. Text content matching
6. Force click for elements with overlays
7. Select dropdown handling

This multi-layered fallback approach significantly improves interaction reliability across different websites.

## Custom Output Processing

Browser-Use allows you to customize how the agent's output is processed and validated:

```python
# From examples/features/custom_output.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent

async def process_output(output: str) -> str:
    """Custom function to process the agent's output."""
    # Add a timestamp to the output
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format the output
    processed = f"[{timestamp}] Result: {output}"

    # You could also save to a file, send to an API, etc.
    with open("agent_results.txt", "a") as f:
        f.write(processed + "\n")

    return processed

async def main():
    llm = ChatOpenAI(model='gpt-4o')
    agent = Agent(
        task="Search for the latest news about artificial intelligence",
        llm=llm,
        process_output=process_output,  # Set the custom output processor
    )

    result = await agent.run()
    print(f"Processed output: {result.final_answer}")

asyncio.run(main())
```

## Output Validation

You can enable built-in output validation to ensure high-quality results:

```python
# From examples/features/validate_output.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent

async def main():
    llm = ChatOpenAI(model='gpt-4o')
    agent = Agent(
        task="Find the definition of 'machine learning' and summarize it in exactly 3 bullet points",
        llm=llm,
        validate_output=True,  # Enable output validation
    )

    result = await agent.run()
    print(result.final_answer)

asyncio.run(main())
```

The validation process:

1. The agent completes its task and generates an output
2. A validation prompt is sent to the LLM with the output and original task
3. The LLM evaluates if the output fully satisfies the task requirements
4. If validation fails, the agent continues working to improve the result
5. This process repeats until a satisfactory output is produced

## Initial Actions

Browser-Use supports specifying initial actions that are performed before the agent starts reasoning:

```python
# From examples/features/initial_actions.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent

async def main():
    llm = ChatOpenAI(model='gpt-4o')
    agent = Agent(
        task="Find information about the latest iPhone model",
        llm=llm,
        initial_actions=["goto:https://www.apple.com/iphone/"],  # Start directly on the iPhone page
    )

    await agent.run()

asyncio.run(main())
```

Initial actions can include:

- `goto:URL` - Navigate to a specific URL
- `click:selector` - Click an element matching a selector
- `fill:selector:value` - Fill a form field
- `screenshot` - Take a screenshot
- Custom functions defined through your controller

## Follow-Up Tasks

Agent tasks can be chained together, with subsequent tasks informed by previous results:

```python
# From examples/features/follow_up_tasks.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent

async def main():
    llm = ChatOpenAI(model='gpt-4o')
    browser = Browser(config=BrowserConfig(headless=False))

    # First task
    agent1 = Agent(
        task="Find the top 3 bestselling books on Amazon right now",
        llm=llm,
        browser=browser,
    )
    result1 = await agent1.run()

    # Create a follow-up task based on first result
    agent2 = Agent(
        task=f"Using this information about bestselling books: '{result1.final_answer}', find reviews for the first book mentioned",
        llm=llm,
        browser=browser,
    )
    result2 = await agent2.run()

    await browser.close()

asyncio.run(main())
```

## File Downloads

Browser-Use supports downloading files from websites:

```python
# From examples/features/download_file.py
import asyncio
import os
from langchain_openai import ChatOpenAI
from browser_use import Agent, Controller

controller = Controller()

@controller.action("Save downloaded file")
def save_downloaded_file(file_content: str, file_name: str):
    """Save a downloaded file to disk."""
    download_dir = os.path.join(os.getcwd(), "downloads")
    os.makedirs(download_dir, exist_ok=True)

    file_path = os.path.join(download_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(file_content.encode() if isinstance(file_content, str) else file_content)

    return f"File saved to {file_path}"

async def main():
    llm = ChatOpenAI(model='gpt-4o')
    agent = Agent(
        task="Go to https://www.africau.edu/images/default/sample.pdf and download the PDF file",
        llm=llm,
        controller=controller,
    )

    await agent.run()

asyncio.run(main())
```

## Drag and Drop

Browser-Use supports drag and drop operations for complex UI interactions:

```python
# From examples/features/drag_drop.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Controller, ActionResult
from typing import Optional

controller = Controller()

@controller.action("Drag and drop element")
async def drag_and_drop(source_index: int, target_index: Optional[int] = None, browser=None):
    """Drag an element to another element or position."""
    try:
        # Get source element
        source_el = await browser.get_dom_element_by_index(source_index)
        source_handle = await browser.get_locate_element(source_el)

        if target_index is not None:
            # Drag to target element
            target_el = await browser.get_dom_element_by_index(target_index)
            target_handle = await browser.get_locate_element(target_el)

            # Perform drag and drop
            await source_handle.drag_to(target_handle)
            return ActionResult(extracted_content=f"Dragged element {source_index} to element {target_index}")
        else:
            # Drag to a position (e.g., center of viewport)
            page = await browser.get_current_page()
            viewport_size = await page.viewport_size()
            await source_handle.drag_to(
                target=None,
                target_position={"x": viewport_size["width"]/2, "y": viewport_size["height"]/2}
            )
            return ActionResult(extracted_content=f"Dragged element {source_index} to center of viewport")
    except Exception as e:
        return ActionResult(error=f"Drag and drop failed: {str(e)}")

async def main():
    llm = ChatOpenAI(model='gpt-4o')
    agent = Agent(
        task="Go to https://the-internet.herokuapp.com/drag_and_drop and drag box A to position B",
        llm=llm,
        controller=controller,
    )

    await agent.run()

asyncio.run(main())
```

## Security Features

Browser-Use includes features for handling sensitive data and restricting URLs:

```python
# From examples/features/sensitive_data.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent
from pydantic import SecretStr

async def main():
    llm = ChatOpenAI(model='gpt-4o')

    # Using SecretStr to protect sensitive data
    username = SecretStr("my_username")
    password = SecretStr("my_secure_password")

    agent = Agent(
        task=f"Log in to example.com using username {username.get_secret_value()} and password {password.get_secret_value()}",
        llm=llm,
        # The LLM won't see the actual credentials in history
    )

    await agent.run()

asyncio.run(main())
```

URL restrictions can be implemented as well:

```python
# From examples/features/restrict_urls.py
import asyncio
import re
from langchain_openai import ChatOpenAI
from browser_use import Agent

# Define allowed URLs pattern
allowed_domains = [
    r"^https://www\.google\.com",
    r"^https://en\.wikipedia\.org",
]

# URL restriction function
def is_url_allowed(url):
    return any(re.match(pattern, url) for pattern in allowed_domains)

async def main():
    llm = ChatOpenAI(model='gpt-4o')
    agent = Agent(
        task="Find information about quantum computing",
        llm=llm,
        restrict_urls=is_url_allowed,  # Only allow Google and Wikipedia
    )

    await agent.run()

asyncio.run(main())
```

## State Management

Browser-Use provides tools for saving and loading agent state:

```python
# From examples/features/save_trace.py
import asyncio
import json
from langchain_openai import ChatOpenAI
from browser_use import Agent

async def main():
    llm = ChatOpenAI(model='gpt-4o')
    agent = Agent(
        task="Search for the latest news about climate change",
        llm=llm,
    )

    # Run the agent
    result = await agent.run()

    # Save trace to file
    with open("agent_trace.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # You can also use the built-in method
    result.save_to_file("agent_history.json")

asyncio.run(main())
```

## Memory Management with Task Persistence

Browser-Use supports maintaining memory across multiple agent runs:

```python
# From examples/features/task_with_memory.py
import asyncio
from browser_use import Agent, AgentMemory
from langchain_openai import ChatOpenAI

async def main():
    # Create a persistent memory object
    memory = AgentMemory()

    # First agent run with memory
    agent1 = Agent(
        task="Search for the tallest building in the world and note its height",
        llm=ChatOpenAI(model='gpt-4o'),
        memory=memory,  # Pass the memory object
    )
    result1 = await agent1.run()

    # Second agent with the same memory
    agent2 = Agent(
        task="Now find the second tallest building and compare its height to the first one you found",
        llm=ChatOpenAI(model='gpt-4o'),
        memory=memory,  # Same memory object contains context from first run
    )
    result2 = await agent2.run()

asyncio.run(main())
```

## Conclusion

The advanced features in Browser-Use significantly extend its capabilities beyond basic browser automation. By leveraging these features, you can create sophisticated web agents that handle complex interactions, manage state, process outputs, and integrate with external systems.

In the next chapter, we'll explore creating multi-agent systems with Browser-Use.
