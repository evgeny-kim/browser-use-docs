# 4. Basic Usage Patterns

This chapter explores common patterns for using Browser-Use, including different approaches to task execution, LLM integration, and browser automation.

## Task Execution

Tasks in Browser-Use are defined as natural language instructions that describe what you want the agent to accomplish. Tasks can range from simple web searches to complex multi-step processes.

### Simple Task Execution

The most basic pattern is executing a single task:

```python
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use import Agent

load_dotenv()

async def main():
    agent = Agent(
        task="Search for the latest news about artificial intelligence",
        llm=ChatOpenAI(model="gpt-4o"),
    )

    result = await agent.run()
    print(result.final_answer)

if __name__ == "__main__":
    asyncio.run(main())
```

### Task with Step Limit

For tasks where you want to limit the execution steps:

```python
# Run with a maximum of 10 steps
result = await agent.run(max_steps=10)
```

### Sequential Tasks

For running multiple tasks in sequence:

```python
async def main():
    # Create a browser instance to reuse
    browser = Browser(config=BrowserConfig(headless=False, keep_alive=True))
    llm = ChatOpenAI(model="gpt-4o")

    # First task
    agent1 = Agent(
        task="Search for recent advances in renewable energy",
        llm=llm,
        browser=browser,
    )
    result1 = await agent1.run()

    # Second task (using same browser)
    agent2 = Agent(
        task="Find the top 3 companies working on solar energy storage",
        llm=llm,
        browser=browser,
    )
    result2 = await agent2.run()

    # Clean up
    await browser.close()
```

## LLM Integration

Browser-Use integrates with various LLM providers through the LangChain library, making it easy to use different models.

### OpenAI Models

```python
from langchain_openai import ChatOpenAI

# Using GPT-4o
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# Using GPT-3.5 Turbo
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

agent = Agent(task="Your task here", llm=llm)
```

### Google Gemini Models

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

# Using Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    api_key=SecretStr(os.getenv("GEMINI_API_KEY"))
)

agent = Agent(task="Your task here", llm=llm)
```

### Claude Models

```python
from langchain_anthropic import ChatAnthropic

# Using Claude
llm = ChatAnthropic(model="claude-3-7-sonnet-20240307")

agent = Agent(task="Your task here", llm=llm)
```

## Browser Automation Patterns

Browser-Use offers several patterns for browser automation, from simple to advanced use cases.

### Default Browser

By default, the Agent class creates a browser instance automatically:

```python
agent = Agent(task="Your task here", llm=your_llm)
# Browser created automatically
```

### Custom Browser Configuration

For more control, create a custom browser instance:

```python
from browser_use import Browser, BrowserConfig, BrowserContextConfig

browser = Browser(
    config=BrowserConfig(
        headless=False,  # Show the browser window
        disable_security=False,  # Keep security features enabled
        stealth_mode=True,  # Enable anti-detection measures
        new_context_config=BrowserContextConfig(
            browser_window_size={"width": 1280, "height": 800},
            minimum_wait_page_load_time=1.0,
            maximum_wait_page_load_time=10.0
        )
    )
)

agent = Agent(task="Your task here", llm=your_llm, browser=browser)
```

### Stealth Mode

For websites with bot detection, use stealth mode:

```python
browser = Browser(
    config=BrowserConfig(
        headless=False,
        stealth_mode=True,  # Enable anti-bot detection features
    )
)
```

Example from stealth.py:

```python
async def main():
    browser = Browser(
        config=BrowserConfig(
            headless=False,
            disable_security=False,
            keep_alive=True,
            new_context_config=BrowserContextConfig(
                keep_alive=True,
                disable_security=False,
            ),
        )
    )

    agent = Agent(
        task="Go to https://bot-detector.rebrowser.net/ and verify that all the bot checks are passed.",
        llm=ChatOpenAI(model='gpt-4o'),
        browser=browser,
    )
    await agent.run()
```

### Using CDP (Chrome DevTools Protocol)

You can connect to an existing Chrome instance using CDP:

```python
browser = Browser(
    config=BrowserConfig(
        headless=False,
        cdp_url='http://localhost:9222',  # Connect to running Chrome instance
    )
)
```

Example from using_cdp.py:

```python
"""
To use CDP:
1. Create a shortcut for Chrome
2. Add --remote-debugging-port=9222 to the target
3. Launch Chrome using this shortcut
4. Verify CDP is running at http://localhost:9222/json/version
"""

browser = Browser(
    config=BrowserConfig(
        headless=False,
        cdp_url='http://localhost:9222',
    )
)

async def main():
    task = 'In docs.google.com write my Papa a quick thank you letter'
    task += ' and save the document as pdf'
    model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

    agent = Agent(
        task=task,
        llm=model,
        browser=browser,
    )

    await agent.run()
    await browser.close()
```

## Handling Browser Resources

It's important to properly manage browser resources to avoid memory leaks:

### Explicit Browser Cleanup

```python
browser = Browser(config=BrowserConfig())
agent = Agent(task="Your task", llm=your_llm, browser=browser)

await agent.run()
await browser.close()  # Explicitly close the browser when done
```

### Browser Reuse

For multiple tasks, you can reuse the same browser:

```python
browser = Browser(config=BrowserConfig(keep_alive=True))

# First agent using the browser
agent1 = Agent(task="First task", llm=your_llm, browser=browser)
await agent1.run()

# Second agent using the same browser
agent2 = Agent(task="Second task", llm=your_llm, browser=browser)
await agent2.run()

# Close when completely done
await browser.close()
```

## Task Design Patterns

How you phrase tasks can significantly impact the agent's performance:

### Clear and Specific Tasks

```python
# Good: Clear and specific
task = "Go to arxiv.org, search for papers on 'large language models' published in 2023, and summarize the top 3 results"

# Less effective: Vague and ambiguous
task = "Find some information about language models"
```

### Multi-Step Tasks

For complex operations, break them down in the task description:

```python
task = """
1. Go to LinkedIn.com
2. Search for 'machine learning engineer' jobs
3. Filter for 'Remote' jobs
4. Extract the titles, companies, and links for the top 5 positions
5. Create a formatted summary of these jobs
"""
```

### Incorporating Context

Include relevant context in the task:

```python
task = """
I'm researching renewable energy technologies for a project.
Go to Google Scholar and find the most cited papers on solar panel efficiency from the last 2 years.
Extract the titles, authors, publication dates, and citation counts for the top 5 papers.
"""
```

## Conclusion

This chapter covered the basic usage patterns for Browser-Use, including task execution, LLM integration, and browser automation. By understanding these patterns, you can effectively implement browser automation for a wide range of use cases.

In the next chapter, we'll explore various use cases and examples to demonstrate Browser-Use in real-world scenarios.
