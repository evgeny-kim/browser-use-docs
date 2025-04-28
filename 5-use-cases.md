# 5. Use Cases and Examples

This chapter explores practical applications of Browser-Use through various examples. We'll examine real-world scenarios where browser automation with LLMs can solve complex tasks.

## Web Navigation and Search

Browser-Use excels at general web navigation tasks, allowing agents to browse websites, search for information, and extract relevant data.

### Web Voyager Agent

The Web Voyager Agent is a general-purpose navigation agent that can handle complex tasks like flight booking, hotel reservations, and course searching.

```python
# From examples/use-cases/web_voyager_agent.py
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig

# Load environment variables
load_dotenv()

# Initialize browser with specific configuration
browser = Browser(
    config=BrowserConfig(
        headless=False,
        disable_security=True,
        new_context_config=BrowserContextConfig(
            disable_security=True,
            minimum_wait_page_load_time=1,
            maximum_wait_page_load_time=10,
            browser_window_size={
                'width': 1280,
                'height': 1100,
            },
        ),
    )
)

# Configure the LLM
llm = AzureChatOpenAI(
    model='gpt-4o',
    api_version='2024-10-21',
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
    api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
)

# Example tasks
HOTEL_BOOKING_TASK = """
Find and book a hotel in Paris with suitable accommodations for a family of four
(two adults and two children) offering free cancellation for the dates of
February 14-21, 2025. on https://www.booking.com/
"""

FLIGHT_SEARCH_TASK = """
Find the lowest-priced one-way flight from Cairo to Montreal on
February 21, 2025, including the total travel time and number of stops.
on https://www.google.com/travel/flights/
"""

COURSE_SEARCH_TASK = """
Browse Coursera, which universities offer Master of Advanced Study in
Engineering degrees? Tell me what is the latest application deadline
for this degree? on https://www.coursera.org/
"""

async def main():
    agent = Agent(
        task=HOTEL_BOOKING_TASK,  # Choose one of the tasks
        llm=llm,
        browser=browser,
        validate_output=True,  # Validates the agent's final output
    )
    history = await agent.run(max_steps=50)
    history.save_to_file('./tmp/history.json')  # Save interaction history

if __name__ == '__main__':
    asyncio.run(main())
```

This example demonstrates how to create an agent for complex web navigation tasks with:

- Custom browser configuration for better UI interaction
- Step limitations to prevent infinite loops
- Output validation for higher quality results
- History saving for later analysis

## Form Filling and Automation

Browser-Use can automate form filling, job applications, and other data entry tasks.

### Job Search and Application

This example demonstrates how to search for jobs, evaluate their relevance based on a CV, and handle job applications:

```python
# From examples/use-cases/find_and_apply_to_jobs.py
import asyncio
import csv
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, SecretStr
from PyPDF2 import PdfReader

from browser_use import ActionResult, Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig

# Path to resume/CV
CV = Path.cwd() / 'cv_04_24.pdf'

# Define a data model for jobs
class Job(BaseModel):
    title: str
    link: str
    company: str
    fit_score: float
    location: Optional[str] = None
    salary: Optional[str] = None

# Create a controller for custom actions
controller = Controller()

# Custom action to save jobs to a file
@controller.action('Save jobs to file - with a score how well it fits to my profile', param_model=Job)
def save_jobs(job: Job):
    with open('jobs.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([job.title, job.company, job.link, job.salary, job.location])
    return 'Saved job to file'

# Custom action to read the CV
@controller.action('Read my cv for context to fill forms')
def read_cv():
    pdf = PdfReader(CV)
    text = ''
    for page in pdf.pages:
        text += page.extract_text() or ''
    return ActionResult(extracted_content=text, include_in_memory=True)

# Custom action to upload CV to application forms
@controller.action('Upload cv to element')
async def upload_cv(index: int, browser: BrowserContext):
    path = str(CV.absolute())
    dom_el = await browser.get_dom_element_by_index(index)

    if dom_el is None:
        return ActionResult(error=f'No element found at index {index}')

    file_upload_dom_el = dom_el.get_file_upload_element()
    file_upload_el = await browser.get_locate_element(file_upload_dom_el)

    try:
        await file_upload_el.set_input_files(path)
        return ActionResult(extracted_content=f'Successfully uploaded CV to index {index}')
    except Exception as e:
        return ActionResult(error=f'Failed to upload file: {str(e)}')

# Main function to run job search and application
async def main():
    # Job search task for Google
    task = (
        'You are a professional job finder. '
        '1. Read my cv with read_cv '
        '2. Find ML internships at Google and save them to a file '
        '3. Rate each job with a fit score based on my CV'
    )

    model = AzureChatOpenAI(
        model='gpt-4o',
        api_version='2024-10-21',
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
        api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
    )

    browser = Browser(config=BrowserConfig(headless=False))
    agent = Agent(
        task=task,
        llm=model,
        controller=controller,
        browser=browser
    )

    await agent.run()

if __name__ == '__main__':
    asyncio.run(main())
```

This example showcases:

- Custom controller actions for specialized tasks
- File uploads and PDF processing
- Data model definitions for structured information
- CSV file output for data persistence

## Handling CAPTCHAs

Browser-Use can also handle CAPTCHAs, which are often a barrier to web automation.

### CAPTCHA Solver

```python
# From examples/use-cases/captcha.py
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent

async def main():
    load_dotenv()

    llm = ChatOpenAI(model='gpt-4o')
    agent = Agent(
        task='go to https://captcha.com/demos/features/captcha-demo.aspx and solve the captcha',
        llm=llm,
    )

    await agent.run()

if __name__ == '__main__':
    asyncio.run(main())
```

This example demonstrates:

- Visual CAPTCHA solving capabilities
- Simple agent configuration for specialized tasks
- LLM's visual reasoning abilities

## E-commerce and Shopping

Browser-Use can automate shopping tasks, product searches, and comparison across e-commerce sites.

### Product Shopping

```python
# From examples/use-cases/shopping.py
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent

async def main():
    load_dotenv()

    llm = ChatOpenAI(model='gpt-4o')
    task = """
    I need a new smartphone with a budget of $800. Go to Amazon.com and:
    1. Search for smartphones under $800
    2. Compare at least 3 options based on:
       - Camera quality
       - Battery life
       - Screen size
       - Storage
       - Customer reviews
    3. Recommend the best option and explain why
    """

    agent = Agent(task=task, llm=llm)
    result = await agent.run()
    print(result.final_answer)

if __name__ == '__main__':
    asyncio.run(main())
```

## Social Media Automation

Browser-Use can automate social media interactions, from posting content to analyzing trends.

### Twitter/X Posting

```python
# From examples/use-cases/post-twitter.py
import asyncio
import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use import ActionResult, Agent, Controller

# Custom data model for tweets
class Tweet(BaseModel):
    content: str
    hashtags: Optional[List[str]] = None
    mentions: Optional[List[str]] = None
    reply_to_url: Optional[str] = None

# Create controller for custom actions
controller = Controller()

# Custom action to create tweet
@controller.action("Create a tweet", param_model=Tweet)
async def create_tweet(tweet: Tweet, agent):
    browser = agent.browser_context

    # Navigate to Twitter
    await browser.goto("https://twitter.com/home")
    time.sleep(3)

    # Click tweet button
    tweet_buttons = await browser.query_selector_all("div[role='button'][data-testid='tweetButtonInline']")
    if not tweet_buttons:
        return ActionResult(error="Could not find tweet button")

    await tweet_buttons[0].click()
    time.sleep(1)

    # Enter tweet text
    tweet_text = tweet.content
    if tweet.hashtags:
        tweet_text += " " + " ".join([f"#{tag}" for tag in tweet.hashtags])
    if tweet.mentions:
        tweet_text += " " + " ".join([f"@{mention}" for mention in tweet.mentions])

    editor = await browser.query_selector("div[role='textbox'][data-testid='tweetTextarea_0']")
    await editor.fill(tweet_text)

    # Submit tweet
    post_button = await browser.query_selector("div[role='button'][data-testid='tweetButton']")
    await post_button.click()
    time.sleep(3)

    return ActionResult(extracted_content="Tweet posted successfully")

async def main():
    load_dotenv()

    llm = ChatOpenAI(model='gpt-4o')
    task = """
    Create and post a tweet about the latest developments in AI and browser automation.
    Include relevant hashtags and mention @OpenAI.
    """

    agent = Agent(task=task, llm=llm, controller=controller)
    await agent.run()

if __name__ == '__main__':
    asyncio.run(main())
```

## Research and Data Collection

Browser-Use enables automated research, data collection, and content extraction from websites.

### Scientific Paper Search

```python
# From examples/use-cases/wikipedia_banana_to_quantum.py
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent

async def main():
    load_dotenv()

    llm = ChatOpenAI(model='gpt-4o')
    task = """
    Start on the Wikipedia page for "Banana" and navigate to the Wikipedia
    page for "Quantum mechanics" by clicking only on internal Wikipedia links.
    Keep track of the path you take.
    """

    agent = Agent(task=task, llm=llm)
    result = await agent.run()
    print(result.final_answer)

if __name__ == '__main__':
    asyncio.run(main())
```

## Automated Testing and Monitoring

Browser-Use can be used for website testing, monitoring, and validation.

### Appointment Check

```python
# From examples/use-cases/check_appointment.py
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent

async def main():
    load_dotenv()

    llm = ChatOpenAI(model='gpt-4o')
    task = """
    Check for available visa appointment slots on the Greece MFA website.
    Go to https://www.visa-gr.net/ and:
    1. Select English language
    2. Navigate to the appointment system
    3. Check if there are any available slots for the next 3 months
    4. Report the earliest available date if any
    """

    agent = Agent(task=task, llm=llm)
    result = await agent.run()
    print(result.final_answer)

if __name__ == '__main__':
    asyncio.run(main())
```

## Scroll and Extract

Browser-Use can handle infinite scrolling pages and extract content while scrolling.

```python
# From examples/use-cases/scrolling_page.py
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent

async def main():
    load_dotenv()

    llm = ChatOpenAI(model='gpt-4o')
    task = """
    Go to Twitter/X and scroll through Elon Musk's profile (@elonmusk).
    Collect his last 10 tweets and summarize the main topics he's discussing.
    Look for any tweets related to AI or SpaceX.
    """

    agent = Agent(task=task, llm=llm)
    result = await agent.run()
    print(result.final_answer)

if __name__ == '__main__':
    asyncio.run(main())
```

## Multi-Agent Collaboration

Browser-Use supports multi-agent systems where agents can collaborate on complex tasks.

```python
# From examples/use-cases/online_coding_agent.py
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent, Browser, BrowserConfig

async def main():
    load_dotenv()

    # Shared browser for both agents
    browser = Browser(config=BrowserConfig(headless=False))
    llm = ChatOpenAI(model='gpt-4o')

    # First agent: code writer
    coder_task = """
    Go to https://replit.com/ and log in as a guest.
    Create a new Python repl and write a function that:
    1. Takes a list of integers as input
    2. Returns the list sorted in descending order
    3. Removes any duplicate numbers
    Share the code with the executor agent.
    """

    coder_agent = Agent(
        task=coder_task,
        llm=llm,
        browser=browser
    )

    # Second agent: code executor
    executor_task = """
    Wait for the coder agent to finish writing code.
    Then, run the code with the following test cases:
    1. [1, 5, 3, 2, 5, 1, 7]
    2. [10, 10, 20, 30, 30, 40]
    3. []
    Report the outputs for each test case.
    """

    executor_agent = Agent(
        task=executor_task,
        llm=llm,
        browser=browser
    )

    # Run agents sequentially
    await coder_agent.run()
    await executor_agent.run()
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
```

## Conclusion

These examples demonstrate the versatility of Browser-Use for a wide range of web automation tasks. The library's flexibility allows it to be applied to virtually any scenario requiring browser interaction, from simple web searches to complex multi-step workflows.

Key strengths demonstrated across these examples include:

- Natural language task definition
- Custom action definition for specialized tasks
- Form filling and data extraction
- Visual processing (CAPTCHAs, images)
- Data modeling and structure
- Multi-agent collaboration

In the next chapter, we'll explore custom functions and extensions that allow you to extend Browser-Use's capabilities even further.
