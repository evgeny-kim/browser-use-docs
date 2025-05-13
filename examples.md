# Basic Examples

## Simple

Find the cheapest flight between two cities using Kayak.

```python
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use import Agent

load_dotenv()

# Initialize the model
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0.0,
)
task = 'Go to kayak.com and find the cheapest flight from Zurich to San Francisco on 2025-05-01'

agent = Agent(task=task, llm=llm)

async def main():
    await agent.run()

if __name__ == '__main__':
    asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/simple.py)

# Browser

## Real Browser

Configure and use a real Chrome browser instance for an agent by specifying the browser binary path.

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig

browser = Browser(
	config=BrowserConfig(
		browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
	)
)

async def main():
	agent = Agent(
		task='In docs.google.com write my Papa a quick letter',
		llm=ChatOpenAI(model='gpt-4o'),
		browser=browser,
	)

	await agent.run()
	await browser.close()

if __name__ == '__main__':
	asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/browser/real_browser.py)

## Stealth

Configure browser settings to avoid bot detection on websites.

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig

llm = ChatOpenAI(model='gpt-4o')
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

async def main():
	agent = Agent(
		task="Go to https://bot-detector.rebrowser.net/ and verify that all the bot checks are passed.",
		llm=llm,
		browser=browser,
	)
	await agent.run()
	# ... more tasks ...

if __name__ == '__main__':
	asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/browser/stealth.py)

## Using Cdp

Connect to a running Chrome instance using the Chrome DevTools Protocol (CDP) for automation.

```python
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig

load_dotenv()

browser = Browser(
	config=BrowserConfig(
		headless=False,
		cdp_url='http://localhost:9222',
	)
)
controller = Controller()

async def main():
	task = 'In docs.google.com write my Papa a quick thank you for everything letter'
	model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')
	agent = Agent(
		task=task,
		llm=model,
		controller=controller,
		browser=browser,
	)

	await agent.run()
	await browser.close()

if __name__ == '__main__':
	asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/browser/using_cdp.py)

# Custom Functions

## Action Filters

Limit the availability of custom actions to specific domains or pages based on URL filters.

```python
import asyncio
from playwright.async_api import Page
from browser_use.agent.service import Agent, Browser, BrowserContext, Controller

# Initialize controller and registry
controller = Controller()
registry = controller.registry

# Action will only be available to Agent on Google domains because of the domain filter
@registry.action(description='Trigger disco mode', domains=['google.com', '*.google.com'])
async def disco_mode(browser: BrowserContext):
    page = await browser.get_current_page()
    await page.evaluate("""() => {
        document.styleSheets[0].insertRule('@keyframes wiggle { 0% { transform: rotate(0deg); } 50% { transform: rotate(10deg); } 100% { transform: rotate(0deg); } }');

        document.querySelectorAll("*").forEach(element => {
            element.style.animation = "wiggle 0.5s infinite";
        });
    }""")

# Create a custom page filter function that determines if the action should be available
def is_login_page(page: Page) -> bool:
    return 'login' in page.url.lower() or 'signin' in page.url.lower()

# Use the page filter to limit the action to only be available on login pages
@registry.action(description='Use the force, luke', page_filter=is_login_page)
async def use_the_force(browser: BrowserContext):
    # This will only ever run on pages that matched the filter
    page = await browser.get_current_page()
    assert is_login_page(page)

    await page.evaluate("""() => { document.querySelector('body').innerHTML = 'These are not the droids you are looking for';}""")

async def main():
    browser = Browser()
    agent = Agent(
        task="""
            Go to apple.com and trigger disco mode (if don't know how to do that, then just move on).
            Then go to google.com and trigger disco mode.
            After that, go to the Google login page and Use the force, luke.
        """,
        llm=ChatOpenAI(model='gpt-4o'),
        browser=browser,
        controller=controller,
    )

    await agent.run(max_steps=10)
    await browser.close()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/custom-functions/action_filters.py)

## Advanced Search

Implement a custom web search action using an external API (e.g., AskTessa) and process its results.

```python
from pydantic import BaseModel
from browser_use import ActionResult, Agent, Controller
import httpx

class Person(BaseModel):
    name: str
    email: str | None = None

class PersonList(BaseModel):
    people: list[Person]

controller = Controller(exclude_actions=['search_google'], output_model=PersonList)
BEARER_TOKEN = os.getenv('BEARER_TOKEN')

@controller.registry.action('Search the web for a specific query')
async def search_web(query: str):
    keys_to_use = ['url', 'title', 'content', 'author', 'score']
    headers = {'Authorization': f'Bearer {BEARER_TOKEN}'}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'https://asktessa.ai/api/search',
            headers=headers,
            json={'query': query}
        )

    final_results = [
        {key: source[key] for key in keys_to_use if key in source}
        for source in response.json()['sources']
        if source['score'] >= 0.8
    ]
    result_text = json.dumps(final_results, indent=4)
    return ActionResult(extracted_content=result_text, include_in_memory=True)

names = [
    'Ruedi Aebersold',
    'Bernd Bodenmiller',
    # ... more names ...
]

async def main():
    task = 'use search_web with "find email address of the following ETH professor:" for each of the following persons in a list of actions. Finally return the list with name and email if provided'
    task += '\n' + '\n'.join(names)
    model = ChatOpenAI(model='gpt-4o')
    agent = Agent(task=task, llm=model, controller=controller, max_actions_per_step=20)

    history = await agent.run()

    result = history.final_result()
    if result:
        parsed: PersonList = PersonList.model_validate_json(result)

        for person in parsed.people:
            print(f'{person.name} - {person.email}')
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/custom-functions/advanced_search.py)

## Clipboard

Define custom actions to copy text to and paste text from the system clipboard.

```python
import pyperclip
from browser_use import Agent, Controller
from browser_use.agent.views import ActionResult

controller = Controller()

@controller.registry.action('Copy text to clipboard')
def copy_to_clipboard(text: str):
    pyperclip.copy(text)
    return ActionResult(extracted_content=text)

@controller.registry.action('Paste text from clipboard')
async def paste_from_clipboard(browser: BrowserContext):
    text = pyperclip.paste()
    # send text to browser
    page = await browser.get_current_page()
    await page.keyboard.type(text)
    return ActionResult(extracted_content=text)

async def main():
    task = 'Copy the text "Hello, world!" to the clipboard, then go to google.com and paste the text'
    agent = Agent(
        task=task,
        llm=ChatOpenAI(model='gpt-4o'),
        controller=controller,
        browser=browser,
    )
    await agent.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/custom-functions/clipboard.py)

## Custom Hooks Before After Step

Record browser activity and agent state at each step using custom hook functions that interact with an external API.

```python
import asyncio
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pyobjtojson import obj_to_json
from browser_use import Agent

async def record_activity(agent_obj):
    print('--- ON_STEP_START HOOK ---')
    website_html: str = await agent_obj.browser_context.get_page_html()
    website_screenshot: str = await agent_obj.browser_context.take_screenshot()

    # Collect data from agent history
    if hasattr(agent_obj, 'state'):
        history = agent_obj.state.history
    else:
        history = None

    model_thoughts = obj_to_json(obj=history.model_thoughts(), check_circular=False)
    # ... more data collection ...

    model_step_summary = {
        'website_html': website_html,
        'website_screenshot': website_screenshot,
        'url': urls_json_last_elem,
        'model_thoughts': model_thoughts_last_elem,
        'model_outputs': model_outputs_json_last_elem,
        'model_actions': model_actions_json_last_elem,
        'extracted_content': extracted_content_json_last_elem,
    }

    # Send data to API
    send_agent_history_step(data=model_step_summary)

agent = Agent(
    task='Compare the price of gpt-4o and DeepSeek-V3',
    llm=ChatOpenAI(model='gpt-4o'),
)

async def run_agent():
    try:
        await agent.run(on_step_start=record_activity, max_steps=30)
    except Exception as e:
        print(e)

asyncio.run(run_agent())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/custom-functions/custom_hooks_before_after_step.py)

## File Upload

Create custom actions to upload local files to a webpage and read file content within the agent's workflow.

```python
import os
from pathlib import Path
from browser_use import Agent, Controller
from browser_use.agent.views import ActionResult
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext

browser = Browser(
    config=BrowserConfig(
        headless=False,
    )
)
controller = Controller()

@controller.action('Upload file to interactive element with file path ')
async def upload_file(index: int, path: str, browser: BrowserContext, available_file_paths: list[str]):
    if path not in available_file_paths:
        return ActionResult(error=f'File path {path} is not available')

    if not os.path.exists(path):
        return ActionResult(error=f'File {path} does not exist')

    dom_el = await browser.get_dom_element_by_index(index)
    file_upload_dom_el = dom_el.get_file_upload_element()

    if file_upload_dom_el is None:
        return ActionResult(error=f'No file upload element found at index {index}')

    file_upload_el = await browser.get_locate_element(file_upload_dom_el)

    try:
        await file_upload_el.set_input_files(path)
        return ActionResult(extracted_content=f'Successfully uploaded file to index {index}', include_in_memory=True)
    except Exception as e:
        return ActionResult(error=f'Failed to upload file to index {index}: {str(e)}')

@controller.action('Read the file content of a file given a path')
async def read_file(path: str, available_file_paths: list[str]):
    if path not in available_file_paths:
        return ActionResult(error=f'File path {path} is not available')

    async with await anyio.open_file(path, 'r') as f:
        content = await f.read()
    return ActionResult(extracted_content=f'File content: {content}', include_in_memory=True)

# Create test files for upload
available_file_paths = [
    str(Path.cwd() / 'tmp.txt'),
    str(Path.cwd() / 'tmp.pdf'),
    str(Path.cwd() / 'tmp.csv')
]

async def main():
    agent = Agent(
        task='Go to website and upload files to fields',
        llm=ChatOpenAI(model='gpt-4o'),
        controller=controller,
        browser=browser,
        available_file_paths=available_file_paths,
    )
    await agent.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/custom-functions/file_upload.py)

## Hover Element

Implement a custom action to simulate hovering over a web element, specified by index, XPath, or CSS selector.

```python
from pydantic import BaseModel
from browser_use import Agent, Controller
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext

class HoverAction(BaseModel):
    index: int | None = None
    xpath: str | None = None
    selector: str | None = None

controller = Controller()

@controller.registry.action(
    'Hover over an element',
    param_model=HoverAction,
)
async def hover_element(params: HoverAction, browser: BrowserContext):
    """
    Hovers over the element specified by its index from the cached selector map or by XPath.
    """
    if params.xpath:
        # Use XPath to locate the element
        element_handle = await browser.get_locate_element_by_xpath(params.xpath)
        if element_handle is None:
            raise Exception(f'Failed to locate element with XPath {params.xpath}')
    elif params.selector:
        # Use CSS selector to locate the element
        element_handle = await browser.get_locate_element_by_css_selector(params.selector)
        if element_handle is None:
            raise Exception(f'Failed to locate element with CSS Selector {params.selector}')
    elif params.index is not None:
        # Use index to locate the element
        element_node = state.selector_map[params.index]
        element_handle = await browser.get_locate_element(element_node)
        if element_handle is None:
            raise Exception(f'Failed to locate element with index {params.index}')
    else:
        raise Exception('Either index or xpath must be provided')

    try:
        await element_handle.hover()
        return ActionResult(extracted_content=f'🖱️ Hovered over element', include_in_memory=True)
    except Exception as e:
        raise Exception(f'Failed to hover over element: {str(e)}')

async def main():
    task = 'Open webpage and hover the element with the css selector #hoverdivpara'
    agent = Agent(
        task=task,
        llm=ChatOpenAI(model='gpt-4o'),
        controller=controller,
    )
    await agent.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/custom-functions/hover_element.py)

## Notification

Define a custom action to send a notification (e.g., an email) when a task is completed.

```python
import yagmail
from browser_use import Agent, Controller
from browser_use.agent.views import ActionResult

controller = Controller()

@controller.registry.action('Done with task ')
async def done(text: str):
    # To send emails use
    # STEP 1: go to https://support.google.com/accounts/answer/185833
    # STEP 2: Create an app password (you can't use here your normal gmail password)
    # STEP 3: Use the app password in the code below for the password
    yag = yagmail.SMTP('your_email@gmail.com', 'your_app_password')
    yag.send(
        to='recipient@example.com',
        subject='Test Email',
        contents=f'result\n: {text}',
    )

    return ActionResult(is_done=True, extracted_content='Email sent!')

async def main():
    task = 'go to brower-use.com and then done'
    agent = Agent(
        task=task,
        llm=ChatOpenAI(model='gpt-4o'),
        controller=controller
    )
    await agent.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/custom-functions/notification.py)

## Onepassword 2fa

Integrate 1Password to fetch 2FA codes for logging into services like Google.

```python
import asyncio
from onepassword.client import Client
from browser_use import ActionResult, Agent, Controller

controller = Controller()

@controller.registry.action('Get 2FA code from 1Password for Google Account', domains=['*.google.com', 'google.com'])
async def get_1password_2fa() -> ActionResult:
    """
    Custom action to retrieve 2FA/MFA code from 1Password using onepassword.client SDK.
    """
    client = await Client.authenticate(
        auth=OP_SERVICE_ACCOUNT_TOKEN,
        integration_name='Browser-Use',
        integration_version='v1.0.0',
    )

    mfa_code = await client.secrets.resolve(f'op://Private/{OP_ITEM_ID}/One-time passcode')

    return ActionResult(extracted_content=mfa_code)

async def main():
    task = 'Go to account.google.com, enter username and password, then if prompted for 2FA code, get 2FA code from 1Password for and enter it'

    model = ChatOpenAI(model='gpt-4o')
    agent = Agent(task=task, llm=model, controller=controller)

    result = await agent.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/custom-functions/onepassword_2fa.py)

## Save To File Hugging Face

Create a custom action to save structured data (e.g., Hugging Face model information) to a local file.

```python
from pydantic import BaseModel
from browser_use import Agent, Controller
from browser_use.agent.views import ActionResult

controller = Controller()

class Model(BaseModel):
	title: str
	url: str
	likes: int
	license: str

class Models(BaseModel):
	models: list[Model]

@controller.action('Save models', param_model=Models)
def save_models(params: Models):
	with open('models.txt', 'a') as f:
		for model in params.models:
			f.write(f'{model.title} ({model.url}): {model.likes} likes, {model.license}\n')

async def main():
	task = 'Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.'

	agent = Agent(
		task=task,
		llm=ChatOpenAI(model='gpt-4o'),
		controller=controller
	)

	await agent.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/custom-functions/save_to_file_hugging_face.py)

## Validate Output

Enforce a specific output structure for custom actions using Pydantic models and validate the agent's adherence to it.

```python
from pydantic import BaseModel
from browser_use import Agent, Controller, ActionResult

controller = Controller()


class DoneResult(BaseModel):
	title: str
	comments: str
	hours_since_start: int


# we overwrite done() in this example to demonstrate the validator
@controller.registry.action('Done with task', param_model=DoneResult)
async def done(params: DoneResult):
	result = ActionResult(is_done=True, extracted_content=params.model_dump_json())
	print(result)
	# NOTE: this is clearly wrong - to demonstrate the validator
	return 'blablabla'


async def main():
	task = 'Go to hackernews hn and give me the top 1 post'
	model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task=task, llm=model, controller=controller, validate_output=True)
	# NOTE: this should fail to demonstrate the validator
	await agent.run(max_steps=5)
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/validate_output.py)

# Features

## Click Fallback Options

Demonstrates robust element clicking by trying various methods (XPath, CSS selector, text) on a test page with custom select dropdowns.

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Controller

controller = Controller()

async def main():
    # Example tasks showing different ways to click elements
    xpath_task = 'Open http://localhost:8000/, click element with the xpath "/html/body/div/div[1]" and then click on Oranges'
    css_selector_task = 'Open http://localhost:8000/, click element with the selector div.select-display and then click on apples'
    text_task = 'Open http://localhost:8000/, click the third element with the text "Select a fruit" and then click on Apples'
    select_task = 'Open http://localhost:8000/, choose the car BMW'
    button_task = 'Open http://localhost:8000/, click on the button'

    llm = ChatOpenAI(model='gpt-4o')

    # Run different agent tasks demonstrating various click methods
    for task in [xpath_task, css_selector_task, text_task, select_task, button_task]:
        agent = Agent(
            task=task,
            llm=llm,
            controller=controller,
        )
        await agent.run()

if __name__ == '__main__':
    asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/click_fallback_options.py)

## Cross Origin Iframes

Interact with elements inside cross-origin iframes.

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig

browser = Browser(
    config=BrowserConfig(
        browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    )
)
controller = Controller()

async def main():
    agent = Agent(
        task='Click "Go cross-site (simple page)" button on https://csreis.github.io/tests/cross-site-iframe.html then tell me the text within',
        llm=ChatOpenAI(model='gpt-4o', temperature=0.0),
        controller=controller,
        browser=browser,
    )

    await agent.run()
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/cross_origin_iframes.py)

## Custom Output

Define a Pydantic model to structure the agent's final output, for example, extracting a list of Hacker News posts.

```python
from pydantic import BaseModel
from browser_use import Agent, Controller

class Post(BaseModel):
	post_title: str
	post_url: str
	num_comments: int
	hours_since_post: int

class Posts(BaseModel):
	posts: list[Post]

controller = Controller(output_model=Posts)

async def main():
	task = 'Go to hackernews show hn and give me the first 5 posts'
	model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task=task, llm=model, controller=controller)

	history = await agent.run()

	result = history.final_result()
	if result:
		parsed: Posts = Posts.model_validate_json(result)

		for post in parsed.posts:
			print(f'Title:            {post.post_title}')
			print(f'URL:              {post.post_url}')
			print(f'Comments:         {post.num_comments}')
			print(f'Hours since post: {post.hours_since_post}')
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/custom_output.py)

## Custom System Prompt

Extend or override the default system prompt to give the agent specific instructions or context.

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent

extend_system_message = (
    'REMEMBER the most important RULE: ALWAYS open first a new tab and go first to url wikipedia.com no matter the task!!!'
)

# or use override_system_message to completely override the system prompt

async def main():
    task = "do google search to find images of Elon Musk's wife"
    model = ChatOpenAI(model='gpt-4o')
    agent = Agent(task=task, llm=model, extend_system_message=extend_system_message)

    print(
        json.dumps(
            agent.message_manager.system_prompt.model_dump(exclude_unset=True),
            indent=4,
        )
    )

    await agent.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/custom_system_prompt.py)

## Custom User Agent

Set a custom user agent string for the browser context.

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig

browser = Browser(
    config=BrowserConfig(
        # browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    )
)

browser_context = BrowserContext(config=BrowserContextConfig(user_agent='foobarfoo'), browser=browser)

agent = Agent(
    task='go to https://whatismyuseragent.com and find the current user agent string',
    llm=ChatOpenAI(model='gpt-4o'),
    browser_context=browser_context,
    use_vision=True,
)

async def main():
    await agent.run()
    await browser_context.close()

if __name__ == '__main__':
    asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/custom_user_agent.py)

## Download File

Download a file from a webpage to a specified local directory.

```python
import os
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig

browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig(save_downloads_path=os.path.join(os.path.expanduser('~'), 'downloads'))
    )
)

async def run_download():
    agent = Agent(
        task='Go to "https://file-examples.com/" and download the smallest doc file.',
        llm=ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp'),
        max_actions_per_step=8,
        use_vision=True,
        browser=browser,
    )
    await agent.run(max_steps=25)
    await browser.close()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/download_file.py)

## Drag Drop

Perform drag-and-drop operations on web elements, such as reordering items in a list or drawing on a canvas.

```python
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent

task = """
Navigate to: https://sortablejs.github.io/Sortable/.
Then scroll down to the first examplw with title "Simple list example".
Drag the element with name "item 1" to below the element with name "item 3".
"""

async def run_search():
    agent = Agent(
        task=task,
        llm=ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp'),
        max_actions_per_step=1,
        use_vision=True,
    )

    await agent.run(max_steps=25)

if __name__ == '__main__':
    asyncio.run(run_search())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/drag_drop.py)

## Follow Up Tasks

Chain multiple tasks together, where a new task can be added and run after the previous one completes.

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig, Controller

browser = Browser(
    config=BrowserConfig(
        browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        new_context_config=BrowserContextConfig(
            keep_alive=True,
        ),
    ),
)
controller = Controller()

task = 'Find the founders of browser-use and draft them a short personalized message'
agent = Agent(task=task, llm=ChatOpenAI(model='gpt-4o'), controller=controller, browser=browser)

async def main():
    await agent.run()

    # new_task = input('Type in a new task: ')
    new_task = 'Find an image of the founders'

    agent.add_new_task(new_task)

    await agent.run()

if __name__ == '__main__':
    asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/follow_up_tasks.py)

## Initial Actions

Specify a sequence of actions for the agent to perform at the very beginning of its run, before processing the main task.

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent

initial_actions = [
	{'open_tab': {'url': 'https://www.google.com'}},
	{'open_tab': {'url': 'https://en.wikipedia.org/wiki/Randomness'}},
	{'scroll_down': {'amount': 1000}},
]
agent = Agent(
	task='What theories are displayed on the page?',
	initial_actions=initial_actions,
	llm=ChatOpenAI(model='gpt-4o'),
)

async def main():
	await agent.run(max_steps=10)

if __name__ == '__main__':
	import asyncio
	asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/initial_actions.py)

## Multi Tab Handling

Manage multiple browser tabs, such as opening several pages and navigating between them.

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent

# video: https://preview.screen.studio/share/clenCmS6
llm = ChatOpenAI(model='gpt-4o')
agent = Agent(
	task='open 3 tabs with elon musk, trump, and steve jobs, then go back to the first and stop',
	llm=llm,
)

async def main():
	await agent.run()

asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/multi-tab_handling.py)

## Multiple Agents Same Browser

Run multiple agents concurrently or sequentially within the same browser instance and context, allowing them to share session state.

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser

async def main():
    # Persist the browser state across agents
    browser = Browser()
    async with await browser.new_context() as context:
        model = ChatOpenAI(model='gpt-4o')
        current_agent = None

        async def get_input():
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: input('Enter task (p: pause current agent, r: resume, b: break): ')
            )

        while True:
            task = await get_input()

            if task.lower() == 'p':
                # Pause the current agent if one exists
                if current_agent:
                    current_agent.pause()
                continue
            elif task.lower() == 'r':
                # Resume the current agent if one exists
                if current_agent:
                    current_agent.resume()
                continue
            elif task.lower() == 'b':
                # Break the current agent's execution if one exists
                if current_agent:
                    current_agent.stop()
                    current_agent = None
                continue

            # If there's a current agent running, pause it before starting new one
            if current_agent:
                current_agent.pause()

            # Create and run new agent with the task
            current_agent = Agent(
                task=task,
                llm=model,
                browser_context=context,
            )

            # Run the agent asynchronously without blocking
            asyncio.create_task(current_agent.run())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/multiple_agents_same_browser.py)

## Outsource State

Persist and load agent state (excluding history) to/from a file, allowing for resumption or transfer of agent progress.

```python
import anyio
from browser_use.agent.views import AgentState
from browser_use import Agent

# Create initial agent state
agent_state = AgentState()

# Use agent with the state
agent = Agent(
    task=task,
    llm=ChatOpenAI(model='gpt-4o'),
    browser=browser,
    browser_context=browser_context,
    injected_agent_state=agent_state,
    page_extraction_llm=ChatOpenAI(model='gpt-4o-mini'),
)

done, valid = await agent.take_step()

# Clear history before saving state
agent_state.history.history = []

# Save state to file
async with await anyio.open_file('agent_state.json', 'w') as f:
    serialized = agent_state.model_dump_json(exclude={'history'})
    await f.write(serialized)

# Load state back from file
async with await anyio.open_file('agent_state.json', 'r') as f:
    loaded_json = await f.read()
    agent_state = AgentState.model_validate_json(loaded_json)
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/outsource_state.py)

## Parallel Agents

Execute multiple agents simultaneously, each performing a different task in its own browser context.

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig

browser = Browser(
    config=BrowserConfig(
        disable_security=True,
        headless=False,
        new_context_config=BrowserContextConfig(save_recording_path='./tmp/recordings'),
    )
)
llm = ChatOpenAI(model='gpt-4o')

async def main():
    agents = [
        Agent(task=task, llm=llm, browser=browser)
        for task in [
            'Search Google for weather in Tokyo',
            'Check Reddit front page title',
            'Look up Bitcoin price on Coinbase',
            'Find NASA image of the day',
            # 'Check top story on CNN',
            # 'Search latest SpaceX launch date',
            # ...
        ]
    ]

    await asyncio.gather(*[agent.run() for agent in agents])

    # Run another agent after parallel agents complete
    agentX = Agent(
        task='Go to apple.com and return the title of the page',
        llm=llm,
        browser=browser,
    )
    await agentX.run()

    await browser.close()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/parallel_agents.py)

## Pause Agent

Control an agent's execution by pausing, resuming, or stopping it through an external interface or thread.

```python
import threading
from langchain_openai import ChatOpenAI
from browser_use import Agent

class AgentController:
    def __init__(self):
        llm = ChatOpenAI(model='gpt-4o')
        self.agent = Agent(
            task='open in one action https://www.google.com, https://www.wikipedia.org, https://www.youtube.com, https://www.github.com, https://amazon.com',
            llm=llm,
        )
        self.running = False

    async def run_agent(self):
        """Run the agent"""
        self.running = True
        await self.agent.run()

    def start(self):
        """Start the agent in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.run_agent())

    def pause(self):
        """Pause the agent"""
        self.agent.pause()

    def resume(self):
        """Resume the agent"""
        self.agent.resume()

    def stop(self):
        """Stop the agent"""
        self.agent.stop()
        self.running = False

async def main():
    controller = AgentController()
    agent_thread = None

    # ... menu code ...

    if choice == '1' and not agent_thread:
        print('Starting agent...')
        agent_thread = threading.Thread(target=controller.start)
        agent_thread.start()

    elif choice == '2':
        print('Pausing agent...')
        controller.pause()

    elif choice == '3':
        print('Resuming agent...')
        controller.resume()

    elif choice == '4':
        print('Stopping agent...')
        controller.stop()
        if agent_thread:
            agent_thread.join()
            agent_thread = None
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/pause_agent.py)

## Planner

Utilize a separate LLM as a planner to break down a complex task into smaller steps for the agent.

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)
planner_llm = ChatOpenAI(
	model='o3-mini',
)
task = 'your task'

agent = Agent(task=task, llm=llm, planner_llm=planner_llm, use_vision_for_planner=False, planner_interval=1)

async def main():
	await agent.run()

if __name__ == '__main__':
	asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/planner.py)

## Playwright Script Generation

Automatically generate a Playwright script based on the agent's actions, which can then be executed independently.

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig

# Define the task for the agent
TASK_DESCRIPTION = """
1. Go to amazon.com
2. Search for 'i7 14700k'
4. If there is an 'Add to Cart' button, open the product page and then click add to cart.
5. the open the shopping cart page /cart button/ go to cart button.
6. Scroll down to the bottom of the cart page.
7. Scroll up to the top of the cart page.
8. Finish the task.
"""

# Define the path where the Playwright script will be saved
SCRIPT_DIR = Path('./playwright_scripts')
SCRIPT_PATH = SCRIPT_DIR / 'playwright_amazon_cart_script.py'

async def main():
    # Initialize the language model
    llm = ChatOpenAI(model='gpt-4.1', temperature=0.0)

    # Configure the browser
    browser_config = BrowserConfig(headless=False)
    browser = Browser(config=browser_config)

    # Configure the agent
    # The 'save_playwright_script_path' argument tells the agent where to save the script
    agent = Agent(
        task=TASK_DESCRIPTION,
        llm=llm,
        browser=browser,
        save_playwright_script_path=str(SCRIPT_PATH),  # Pass the path as a string
    )

    print('Running the agent to generate the Playwright script...')
    history = await agent.run()

    # ... executing the generated script ...
    if SCRIPT_PATH.exists():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(SCRIPT_PATH),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path.cwd(),  # Run from the current working directory
        )

        # ... output streaming code ...

    # Close the browser used by the agent
    if browser:
        await browser.close()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/playwright_script_generation.py)

## Restrict Urls

Confine the agent's browsing activity to a predefined list of allowed domains.

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)
task = "go to google.com and search for openai.com and click on the first link then extract content and scroll down"

allowed_domains = ['google.com']

browser = Browser(
	config=BrowserConfig(
		browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		new_context_config=BrowserContextConfig(
			allowed_domains=allowed_domains,
		),
	),
)

agent = Agent(
	task=task,
	llm=llm,
	browser=browser,
)

async def main():
	await agent.run(max_steps=25)
	# ...
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/restrict_urls.py)

## Result Processing

Access and process various components of the agent's execution history, such as final results, errors, model actions, and thoughts.

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent
from browser_use.agent.views import AgentHistoryList

llm = ChatOpenAI(model='gpt-4o')
agent = Agent(
    task="go to google.com and type 'OpenAI' click search and give me the first url",
    llm=llm,
    browser_context=browser_context,
)
history: AgentHistoryList = await agent.run(max_steps=3)

print('Final Result:')
pprint(history.final_result(), indent=4)

print('\nErrors:')
pprint(history.errors(), indent=4)

# e.g. xPaths the model clicked on
print('\nModel Outputs:')
pprint(history.model_actions(), indent=4)

print('\nThoughts:')
pprint(history.model_thoughts(), indent=4)
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/result_processing.py)

## Save Trace

Record a Playwright trace of the agent's browser interactions for debugging and visualization.

```python
from langchain_openai import ChatOpenAI
from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)

async def main():
    browser = Browser()

    async with await browser.new_context(config=BrowserContextConfig(trace_path='./tmp/traces/')) as context:
        agent = Agent(
            task='Go to hackernews, then go to apple.com and return all titles of open tabs',
            llm=llm,
            browser_context=context,
        )
        await agent.run()

    await browser.close()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/save_trace.py)

## Sensitive Data

Handle sensitive information (e.g., login credentials) by providing them to the agent in a way that they are used but not directly exposed in logs or prompts.

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent

llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0.0,
)
# the model will see x_name and x_password, but never the actual values.
sensitive_data = {'x_name': 'my_x_name', 'x_password': 'my_x_password'}
task = 'go to x.com and login with x_name and x_password then find interesting posts and like them'

agent = Agent(task=task, llm=llm, sensitive_data=sensitive_data)

async def main():
    await agent.run()

if __name__ == '__main__':
    asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/sensitive_data.py)

## Small Model For Extraction

Use a smaller, potentially faster or cheaper, LLM for page content extraction while a more capable LLM handles main task processing.

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)
small_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
task = 'Find the founders of browser-use in ycombinator, extract all links and open the links one by one'
agent = Agent(task=task, llm=llm, page_extraction_llm=small_llm)

async def main():
	await agent.run()

if __name__ == '__main__':
	asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/small_model_for_extraction.py)

## Task With Memory

Enable long-term memory for the agent to retain information across multiple steps while performing a complex, multi-page task like summarizing documentation.

```python
# Define a list of links to process
links = [
    'https://docs.mem0.ai/components/llms/models/litellm',
    'https://docs.mem0.ai/components/llms/models/mistral_AI',
    # ... more links ...
]

class Link(BaseModel):
    url: str
    title: str
    summary: str

class Links(BaseModel):
    links: list[Link]

initial_actions = [
    {'open_tab': {'url': 'https://docs.mem0.ai/'}},
]
controller = Controller(output_model=Links)
task_description = f"""
Visit all the links provided in {links} and summarize the content of the page with url and title.
There are {len(links)} links to visit. Make sure to visit all the links.
Return a json with the following format: [{{url: <url>, title: <title>, summary: <summary>}}].
"""

async def main(max_steps=500):
    config = BrowserConfig(headless=True)
    browser = Browser(config=config)

    agent = Agent(
        task=task_description,
        llm=ChatOpenAI(model='gpt-4o-mini'),
        controller=controller,
        initial_actions=initial_actions,
        enable_memory=True,
        browser=browser,
    )
    history = await agent.run(max_steps=max_steps)
    result = history.final_result()
    parsed_result = []
    if result:
        parsed: Links = Links.model_validate_json(result)
        # ... process and save results ...
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/task_with_memory.py)

## Validate Output

Enforce a specific output structure for custom actions using Pydantic models and validate the agent's adherence to it.

```python
from pydantic import BaseModel
from browser_use import Agent, Controller, ActionResult

controller = Controller()


class DoneResult(BaseModel):
	title: str
	comments: str
	hours_since_start: int


# we overwrite done() in this example to demonstrate the validator
@controller.registry.action('Done with task', param_model=DoneResult)
async def done(params: DoneResult):
	result = ActionResult(is_done=True, extracted_content=params.model_dump_json())
	print(result)
	# NOTE: this is clearly wrong - to demonstrate the validator
	return 'blablabla'


async def main():
	task = 'Go to hackernews hn and give me the top 1 post'
	model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task=task, llm=model, controller=controller, validate_output=True)
	# NOTE: this should fail to demonstrate the validator
	await agent.run(max_steps=5)
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/features/validate_output.py)

# Integrations

## Discord Api

Create a Discord bot that uses Browser Use to perform tasks based on user messages.

```python
from discord.ext import commands
from langchain_core.language_models.chat_models import BaseChatModel
from browser_use import BrowserConfig
from browser_use.agent.service import Agent, Browser

class DiscordBot(commands.Bot):
    def __init__(
        self,
        llm: BaseChatModel,
        prefix: str = '$bu',
        ack: bool = False,
        browser_config: BrowserConfig = BrowserConfig(headless=True),
    ):
        self.llm = llm
        self.prefix = prefix.strip()
        self.ack = ack
        self.browser_config = browser_config

        # Define intents.
        intents = discord.Intents.default()
        intents.message_content = True  # Enable message content intent
        intents.members = True  # Enable members intent for user info

        # Initialize the bot with a command prefix and intents.
        super().__init__(command_prefix='!', intents=intents)

    async def on_message(self, message):
        """Called when a message is received."""
        try:
            if message.author == self.user:  # Ignore the bot's messages
                return
            if message.content.strip().startswith(f'{self.prefix} '):
                if self.ack:
                    await message.reply('Starting browser use task...', mention_author=True)

                try:
                    agent_message = await self.run_agent(message.content.replace(f'{self.prefix} ', '').strip())
                    await message.channel.send(content=f'{agent_message}', reference=message, mention_author=True)
                except Exception as e:
                    await message.channel.send(
                        content=f'Error during task execution: {str(e)}',
                        reference=message,
                        mention_author=True,
                    )
        except Exception as e:
            print(f'Error in message handling: {e}')

    async def run_agent(self, task: str) -> str:
        try:
            browser = Browser(config=self.browser_config)
            agent = Agent(task=(task), llm=self.llm, browser=browser)
            result = await agent.run()

            agent_message = None
            if result.is_done():
                agent_message = result.history[-1].result[0].extracted_content

            if agent_message is None:
                agent_message = 'Oops! Something went wrong while running Browser-Use.'

            return agent_message
        except Exception as e:
            raise Exception(f'Browser-use task failed: {str(e)}')
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/integrations/discord/discord_api.py)

## Discord Example

Run a Discord bot that listens for commands and executes browser automation tasks using an LLM.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import BrowserConfig
from examples.integrations.discord.discord_api import DiscordBot

# Load credentials from environment variables
bot_token = os.getenv('DISCORD_BOT_TOKEN')
api_key = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

bot = DiscordBot(
    llm=llm,  # required; instance of BaseChatModel
    prefix='$bu',  # optional; prefix of messages to trigger browser-use
    ack=True,  # optional; whether to acknowledge task receipt with a message
    browser_config=BrowserConfig(
        headless=False
    ),  # optional; useful for changing headless mode or other browser configs
)

bot.run(
    token=bot_token,  # required; Discord bot token
)
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/integrations/discord/discord_example.py)

## Slack Api

Develop a Slack bot integrated with FastAPI to handle events and execute browser tasks.

```python
from fastapi import FastAPI, Request, Depends
from slack_sdk.web.async_client import AsyncWebClient
from langchain_core.language_models.chat_models import BaseChatModel
from browser_use import BrowserConfig
from browser_use.agent.service import Agent, Browser

class SlackBot:
    def __init__(
        self,
        llm: BaseChatModel,
        bot_token: str,
        signing_secret: str,
        ack: bool = False,
        browser_config: BrowserConfig = BrowserConfig(headless=True),
    ):
        self.llm = llm
        self.ack = ack
        self.browser_config = browser_config
        self.client = AsyncWebClient(token=bot_token)
        self.signature_verifier = SignatureVerifier(signing_secret)
        self.processed_events = set()

    async def handle_event(self, event, event_id):
        # ... event processing ...
        if text and text.startswith('$bu '):
            task = text[len('$bu ') :].strip()
            if self.ack:
                await self.send_message(
                    event['channel'], f'<@{user_id}> Starting browser use task...', thread_ts=event.get('ts')
                )

            try:
                agent_message = await self.run_agent(task)
                await self.send_message(event['channel'], f'<@{user_id}> {agent_message}', thread_ts=event.get('ts'))
            except Exception as e:
                await self.send_message(event['channel'], f'Error during task execution: {str(e)}', thread_ts=event.get('ts'))

    async def run_agent(self, task: str) -> str:
        try:
            browser = Browser(config=self.browser_config)
            agent = Agent(task=task, llm=self.llm, browser=browser)
            result = await agent.run()

            agent_message = None
            if result.is_done():
                agent_message = result.history[-1].result[0].extracted_content

            if agent_message is None:
                agent_message = 'Oops! Something went wrong while running Browser-Use.'

            return agent_message
        except Exception as e:
            return f'Error during task execution: {str(e)}'

# FastAPI endpoint for handling Slack events
@app.post('/slack/events')
async def slack_events(request: Request, slack_bot: Annotated[SlackBot, Depends()]):
    # ... request verification ...
    event_data = await request.json()
    if 'event' in event_data:
        await slack_bot.handle_event(event_data.get('event'), event_data.get('event_id'))
    return {}
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/integrations/slack/slack_api.py)

## Slack Example

Run a Slack bot that uses Browser Use to perform tasks triggered by Slack messages.

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import BrowserConfig
from examples.integrations.slack.slack_api import SlackBot, app

load_dotenv()

# Load credentials from environment variables
bot_token = os.getenv('SLACK_BOT_TOKEN')
signing_secret = os.getenv('SLACK_SIGNING_SECRET')
api_key = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

slack_bot = SlackBot(
    llm=llm,  # required; instance of BaseChatModel
    bot_token=bot_token,  # required; Slack bot token
    signing_secret=signing_secret,  # required; Slack signing secret
    ack=True,  # optional; whether to acknowledge task receipt with a message
    browser_config=BrowserConfig(
        headless=True
    ),  # optional; useful for changing headless mode or other browser configs
)

app.dependency_overrides[SlackBot] = lambda: slack_bot

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('integrations.slack.slack_api:app', host='0.0.0.0', port=3000)
```

# Models

## Azure Openai

Use an Azure OpenAI model (e.g., GPT-4o) as the LLM for the agent.

```python
from langchain_openai import AzureChatOpenAI
from browser_use import Agent

# Retrieve Azure-specific environment variables
azure_openai_api_key = os.getenv('AZURE_OPENAI_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

# Initialize the Azure OpenAI client
llm = AzureChatOpenAI(
    model_name='gpt-4o',
    openai_api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    deployment_name='gpt-4o',
    api_version='2024-08-01-preview',
)

agent = Agent(
    task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
    llm=llm,
    enable_memory=True,
)

async def main():
    await agent.run(max_steps=10)

asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/models/azure_openai.py)

## Bedrock Claude

Utilize an AWS Bedrock model (e.g., Claude Sonnet) as the LLM for the agent to perform web automation tasks.

```python
import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from browser_use import Agent

def get_llm():
	config = Config(retries={'max_attempts': 10, 'mode': 'adaptive'})
	bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1', config=config)

	return ChatBedrockConverse(
		model_id='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
		temperature=0.0,
		max_tokens=None,
		client=bedrock_client,
	)

# Define the task for the agent
task = (
	"Visit cnn.com, navigate to the 'World News' section, and identify the latest headline. "
	'Open the first article and summarize its content in 3-4 sentences.'
	# ... task continues ...
)

llm = get_llm()

agent = Agent(
	task=args.query,
	llm=llm,
	controller=Controller(),
	browser=browser,
	validate_output=True,
)

async def main():
	await agent.run(max_steps=30)
	await browser.close()

asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/models/bedrock_claude.py)

## Claude 3 7 Sonnet

Employ Anthropic's Claude 3.7 Sonnet model as the LLM for the agent.

```python
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from browser_use import Agent

# Load environment variables from .env file
load_dotenv()

llm = ChatAnthropic(model_name='claude-3-7-sonnet-20250219', temperature=0.0, timeout=30, stop=None)

agent = Agent(
	task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
	llm=llm,
)

async def main():
	await agent.run(max_steps=10)

asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/models/claude-3.7-sonnet.py)

## Deepseek

Use a DeepSeek model (e.g., deepseek-chat) via its API as the LLM for the agent.

```python
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr
from browser_use import Agent

load_dotenv()

api_key = os.getenv('DEEPSEEK_API_KEY', '')

async def run_search():
    agent = Agent(
        task=(
            '1. Go to https://www.reddit.com/r/LocalLLaMA '
            "2. Search for 'browser use' in the search bar"
            '3. Click on first result'
            '4. Return the first comment'
        ),
        llm=ChatDeepSeek(
            base_url='https://api.deepseek.com/v1',
            model='deepseek-chat',
            api_key=SecretStr(api_key),
        ),
        use_vision=False,
    )

    await agent.run()

if __name__ == '__main__':
    asyncio.run(run_search())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/models/deepseek.py)

## DeepSeek R1

Utilize the DeepSeek Reasoner (R1) model for complex web automation tasks.

```python
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr
from browser_use import Agent

load_dotenv()

api_key = os.getenv('DEEPSEEK_API_KEY', '')

async def run_search():
    agent = Agent(
        task=('go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result'),
        llm=ChatDeepSeek(
            base_url='https://api.deepseek.com/v1',
            model='deepseek-reasoner',
            api_key=SecretStr(api_key),
        ),
        use_vision=False,
        max_failures=2,
        max_actions_per_step=1,
    )

    await agent.run()

if __name__ == '__main__':
    asyncio.run(run_search())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/models/deepseek-r1.py)

## Gemini

Leverage a Google Gemini model (e.g., gemini-2.0-flash-exp) as the LLM for the agent.

```python
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import Agent, BrowserConfig
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

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
        max_actions_per_step=4,
        browser=browser,
    )

    await agent.run(max_steps=25)

if __name__ == '__main__':
    asyncio.run(run_search())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/models/gemini.py)

## Gpt 4o

Use OpenAI's GPT-4o model as the LLM for the agent.

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent

llm = ChatOpenAI(model='gpt-4o')
agent = Agent(
	task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
	llm=llm,
)


async def main():
	await agent.run(max_steps=10)


asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/models/gpt-4o.py)

## Grok

Employ a Grok model via its API as the LLM for the agent.

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from browser_use import Agent

load_dotenv()

api_key = os.getenv('GROK_API_KEY', '')

async def run_search():
    agent = Agent(
        task=(
            'Go to amazon.com, search for wireless headphones, filter by highest rating, and return the title and price of the first product'
        ),
        llm=ChatOpenAI(
            base_url='https://api.x.ai/v1',
            model='grok-3-beta',
            api_key=SecretStr(api_key),
        ),
        use_vision=False,
    )

    await agent.run()

if __name__ == '__main__':
    asyncio.run(run_search())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/models/grok.py)

## Novita

Use a Novita.ai model (e.g., deepseek-v3) as the LLM for the agent.

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from browser_use import Agent

load_dotenv()
api_key = os.getenv('NOVITA_API_KEY', '')

async def run_search():
    agent = Agent(
        task=(
            'Go to https://www.reddit.com/r/LocalLLaMA, search for "browser use" in the search bar, '
            'click on first result, and return the first comment'
        ),
        llm=ChatOpenAI(
            base_url='https://api.novita.ai/v3/openai',
            model='deepseek/deepseek-v3-0324',
            api_key=SecretStr(api_key),
        ),
        use_vision=False,
    )

    await agent.run()

if __name__ == '__main__':
    asyncio.run(run_search())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/models/novita.py)

## Ollama

Run an agent using a locally hosted LLM through Ollama (e.g., Qwen 2.5).

```python
import asyncio
from langchain_ollama import ChatOllama
from browser_use import Agent

async def run_search():
    agent = Agent(
        task="Search for a 'browser use' post on the r/LocalLLaMA subreddit and open it.",
        llm=ChatOllama(
            model='qwen2.5:32b-instruct-q4_K_M',
            num_ctx=32000,
        ),
    )

    result = await agent.run()
    return result

if __name__ == '__main__':
    asyncio.run(run_search())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/models/ollama.py)

## Qwen

Use a Qwen model via Ollama as the LLM for the agent.

```python
import asyncio
from langchain_ollama import ChatOllama
from browser_use import Agent

async def run_search():
    agent = Agent(
        task=(
            "1. Go to https://www.reddit.com/r/LocalLLaMA2. Search for 'browser use' in the search bar3. Click search4. Call done"
        ),
        llm=ChatOllama(
            # model='qwen2.5:32b-instruct-q4_K_M',
            # model='qwen2.5:14b',
            model='qwen2.5:latest',
            num_ctx=128000,
        ),
        max_actions_per_step=1,
    )

    await agent.run()

if __name__ == '__main__':
    asyncio.run(run_search())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/models/qwen.py)

# UI

## Command Line

Run browser automation tasks from the command line, specifying the query and LLM provider (OpenAI or Anthropic).

```python
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
        return ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.0)
    elif provider == 'openai':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model='gpt-4o', temperature=0.0)
    else:
        raise ValueError(f'Unsupported provider: {provider}')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Automate browser tasks using an LLM agent.')
    parser.add_argument(
        '--query', type=str, help='The query to process', default='go to reddit and search for posts about browser-use'
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
    args = parse_arguments()
    agent, browser = initialize_agent(args.query, args.provider)
    await agent.run(max_steps=25)
    input('Press Enter to close the browser...')
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/ui/command_line.py)

## Gradio Demo

Create a Gradio web interface to input tasks and API keys for running Browser Use agents.

```python
import asyncio
import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use import Agent

load_dotenv()

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

## Streamlit Demo

Build a Streamlit web application to control a Browser Use agent, allowing users to input queries and select LLM providers.

```python
import asyncio
import os
import streamlit as st
from dotenv import load_dotenv
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller

# Load environment variables
load_dotenv()

# Function to get the LLM based on provider
def get_llm(provider: str):
    if provider == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.0)
    elif provider == 'openai':
        from langchain_openai import ChatOpenAI
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

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/ui/streamlit_demo.py)

# Use Cases

## Captcha

Attempt to solve CAPTCHAs on a demo website.

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent

llm = ChatOpenAI(model='gpt-4o')
agent = Agent(
    task='go to https://captcha.com/demos/features/captcha-demo.aspx and solve the captcha',
    llm=llm,
)
await agent.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/use-cases/captcha.py)

## Check Appointment

Check for available visa appointment slots on a government website.

```python
from pydantic import BaseModel
from browser_use.agent.service import Agent
from browser_use.controller.service import Controller

controller = Controller()

class WebpageInfo(BaseModel):
    """Model for webpage link."""
    link: str = 'https://appointment.mfa.gr/en/reservations/aero/ireland-grcon-dub/'

@controller.action('Go to the webpage', param_model=WebpageInfo)
def go_to_webpage(webpage_info: WebpageInfo):
    """Returns the webpage link."""
    return webpage_info.link

async def main():
    task = (
        'Go to the Greece MFA webpage via the link I provided you.'
        'Check the visa appointment dates. If there is no available date in this month, check the next month.'
        'If there is no available date in both months, tell me there is no available date.'
    )

    model = ChatOpenAI(model='gpt-4o-mini')
    agent = Agent(task, model, controller=controller, use_vision=True)

    await agent.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/use-cases/check_appointment.py)

## Find And Apply To Jobs

Automate searching for job listings, evaluating them against a CV, and initiating applications.

```python
from pydantic import BaseModel
from browser_use import ActionResult, Agent, Controller
from PyPDF2 import PdfReader

controller = Controller()
CV = Path.cwd() / 'cv_04_24.pdf'

class Job(BaseModel):
    title: str
    link: str
    company: str
    fit_score: float
    location: str | None = None
    salary: str | None = None

@controller.action('Save jobs to file - with a score how well it fits to my profile', param_model=Job)
def save_jobs(job: Job):
    with open('jobs.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([job.title, job.company, job.link, job.salary, job.location])

    return 'Saved job to file'

@controller.action('Read my cv for context to fill forms')
def read_cv():
    pdf = PdfReader(CV)
    text = ''
    for page in pdf.pages:
        text += page.extract_text() or ''
    return ActionResult(extracted_content=text, include_in_memory=True)

@controller.action(
    'Upload cv to element - call this function to upload if element is not found, try with different index of the same upload element',
)
async def upload_cv(index: int, browser: BrowserContext):
    path = str(CV.absolute())
    dom_el = await browser.get_dom_element_by_index(index)
    # ...
    try:
        await file_upload_el.set_input_files(path)
        msg = f'Successfully uploaded file "{path}" to index {index}'
        return ActionResult(extracted_content=msg)
    except Exception as e:
        return ActionResult(error=f'Failed to upload file to index {index}')

async def main():
    ground_task = (
        'You are a professional job finder. '
        '1. Read my cv with read_cv'
        'find ml internships in and save them to a file'
        'search at company:'
    )
    tasks = [
        ground_task + '\n' + 'Google',
        # ...
    ]

    agents = []
    for task in tasks:
        agent = Agent(task=task, llm=model, controller=controller, browser=browser)
        agents.append(agent)

    await asyncio.gather(*[agent.run() for agent in agents])
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/use-cases/find_and_apply_to_jobs.py)

## Find Influencer Profiles

Extract a username from a TikTok video URL and search the web for associated social media profiles.

```python
from pydantic import BaseModel
import httpx
from browser_use import Agent, Controller
from browser_use.agent.views import ActionResult

class Profile(BaseModel):
    platform: str
    profile_url: str

class Profiles(BaseModel):
    profiles: list[Profile]

controller = Controller(exclude_actions=['search_google'], output_model=Profiles)

@controller.registry.action('Search the web for a specific query')
async def search_web(query: str):
    keys_to_use = ['url', 'title', 'content', 'author', 'score']
    headers = {'Authorization': f'Bearer {BEARER_TOKEN}'}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'https://asktessa.ai/api/search',
            headers=headers,
            json={'query': query},
        )

    final_results = [
        {key: source[key] for key in keys_to_use if key in source}
        for source in await response.json()['sources']
        if source['score'] >= 0.2
    ]
    # ...
    return ActionResult(extracted_content=result_text, include_in_memory=True)

async def main():
    task = (
        'Go to this tiktok video url, open it and extract the @username from the resulting url. Then do a websearch for this username to find all his social media profiles. Return me the links to the social media profiles with the platform name.'
        ' https://www.tiktokv.com/share/video/7470981717659110678/  '
    )
    model = ChatOpenAI(model='gpt-4o')
    agent = Agent(task=task, llm=model, controller=controller)

    history = await agent.run()

    result = history.final_result()
    if result:
        parsed: Profiles = Profiles.model_validate_json(result)

        for profile in parsed.profiles:
            print(f'Platform: {profile.platform}')
            print(f'Profile URL: {profile.profile_url}')
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/use-cases/find_influencer_profiles.py)

## Google Sheets

Automate interactions with Google Sheets, including opening sheets, reading/writing cell data, and clearing ranges.

```python
from browser_use import ActionResult, Agent, Controller
from browser_use.browser.context import BrowserContext

controller = Controller()

def is_google_sheet(page) -> bool:
    return page.url.startswith('https://docs.google.com/spreadsheets/')

@controller.registry.action('Google Sheets: Open a specific Google Sheet')
async def open_google_sheet(browser: BrowserContext, google_sheet_url: str):
    page = await browser.get_current_page()
    if page.url != google_sheet_url:
        await page.goto(google_sheet_url)
        await page.wait_for_load_state()
    if not is_google_sheet(page):
        return ActionResult(error='Failed to open Google Sheet, are you sure you have permissions to access this sheet?')
    return ActionResult(extracted_content=f'Opened Google Sheet {google_sheet_url}', include_in_memory=False)

@controller.registry.action('Google Sheets: Get the contents of a specific cell or range of cells', page_filter=is_google_sheet)
async def get_range_contents(browser: BrowserContext, cell_or_range: str):
    # ...
    await select_cell_or_range(browser, cell_or_range)
    await page.keyboard.press('ControlOrMeta+C')
    extracted_tsv = pyperclip.paste()
    return ActionResult(extracted_content=extracted_tsv, include_in_memory=True)

@controller.registry.action('Google Sheets: Input text into the currently selected cell', page_filter=is_google_sheet)
async def input_selected_cell_text(browser: BrowserContext, text: str):
    page = await browser.get_current_page()
    await page.keyboard.type(text, delay=0.1)
    await page.keyboard.press('Enter')
    # ...
    return ActionResult(extracted_content=f'Inputted text {text}', include_in_memory=False)

async def main():
    async with await browser.new_context() as context:
        model = ChatOpenAI(model='gpt-4o')

        researcher = Agent(
            task="""
                Google to find the full name, nationality, and date of birth of the CEO of the top 10 Fortune 100 companies.
                For each company, append a row to this existing Google Sheet: https://docs.google.com/spreadsheets/d/1INaIcfpYXlMRWO__de61SHFCaqt1lfHlcvtXZPItlpI/edit
                Make sure column headers are present and all existing values in the sheet are formatted correctly.
                Columns:
                    A: Company Name
                    B: CEO Full Name
                    C: CEO Country of Birth
                    D: CEO Date of Birth (YYYY-MM-DD)
                    E: Source URL where the information was found
            """,
            llm=model,
            browser_context=context,
            controller=controller,
        )
        await researcher.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/use-cases/google_sheets.py)

## Online Coding Agent

Implement a multi-agent system where one agent opens an online code editor and another writes and executes code.

```python
from browser_use import Agent, Browser

async def main():
    browser = Browser()
    async with await browser.new_context() as context:
        model = ChatOpenAI(model='gpt-4o')

        # Initialize browser agent
        agent1 = Agent(
            task='Open an online code editor programiz.',
            llm=model,
            browser_context=context,
        )
        executor = Agent(
            task='Executor. Execute the code written by the coder and suggest some updates if there are errors.',
            llm=model,
            browser_context=context,
        )

        coder = Agent(
            task='Coder. Your job is to write and complete code. You are an expert coder. Code a simple calculator. Write the code on the coding interface after agent1 has opened the link.',
            llm=model,
            browser_context=context,
        )
        await agent1.run()
        await executor.run()
        await coder.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/use-cases/online_coding_agent.py)

## Post Twitter

Automate posting new tweets and replying to existing ones on X (Twitter).

```python
from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from dataclasses import dataclass

@dataclass
class TwitterConfig:
    """Configuration for Twitter posting"""
    openai_api_key: str
    chrome_path: str
    target_user: str  # Twitter handle without @
    message: str
    reply_url: str
    headless: bool = False
    model: str = 'gpt-4o-mini'
    base_url: str = 'https://x.com/home'

# Customize these settings
config = TwitterConfig(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    chrome_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    target_user='XXXXX',
    message='XXXXX',
    reply_url='XXXXX',
    headless=False,
)

def create_twitter_agent(config: TwitterConfig) -> Agent:
    llm = ChatOpenAI(model=config.model, api_key=config.openai_api_key)

    browser = Browser(
        config=BrowserConfig(
            headless=config.headless,
            browser_binary_path=config.chrome_path,
        )
    )

    controller = Controller()

    # Construct the full message with tag
    full_message = f'@{config.target_user} {config.message}'

    # Create the agent with detailed instructions
    return Agent(
        task=f"""Navigate to Twitter and create a post and reply to a tweet.

        Here are the specific steps:

        1. Go to {config.base_url}. See the text input field at the top of the page that says "What's happening?"
        2. Look for the text input field at the top of the page that says "What's happening?"
        3. Click the input field and type exactly this message:
        "{full_message}"
        4. Find and click the "Post" button (look for attributes: 'button' and 'data-testid="tweetButton"')
        5. Do not click on the '+' button which will add another tweet.

        6. Navigate to {config.reply_url}
        7. Before replying, understand the context of the tweet by scrolling down and reading the comments.
        8. Reply to the tweet under 50 characters.
        """,
        llm=llm,
        controller=controller,
        browser=browser,
    )

async def main():
    agent = create_twitter_agent(config)
    await agent.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/use-cases/post_twitter.py)

## Scrolling Page

Perform various scrolling actions on a webpage, including scrolling by specific amounts or to a particular text string.

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig

llm = ChatOpenAI(model='gpt-4o')

agent = Agent(
    task="Navigate to 'https://en.wikipedia.org/wiki/Internet' and scroll to the string 'The vast majority of computer'",
    llm=llm,
    browser=Browser(config=BrowserConfig(headless=False)),
)

async def main():
    await agent.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/use-cases/scrolling_page.py)

## Shopping

Automate online grocery shopping, including searching for items, adding to cart, handling substitutions, and proceeding to checkout.

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser

task = """
   ### Prompt for Shopping Agent – Migros Online Grocery Order

Objective:
Visit [Migros Online](https://www.migros.ch/en), search for the required grocery items, add them to the cart, select an appropriate delivery window, and complete the checkout process using TWINT.

Important:
- Make sure that you don't buy more than it's needed for each article.
- After your search, if you click  the "+" button, it adds the item to the basket.
..."""

browser = Browser()

agent = Agent(
    task=task,
    llm=ChatOpenAI(model='gpt-4o'),
    browser=browser,
)

async def main():
    await agent.run()
    input('Press Enter to close the browser...')
    await browser.close()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/use-cases/shopping.py)

## Twitter Post Using Cookies

Post to X (Twitter) by loading authentication cookies from a file to bypass manual login.

```python
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig

# Goal: Automates posting on X (Twitter) using stored authentication cookies.

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

browser = Browser(
	config=BrowserConfig()
)

file_path = os.path.join(os.path.dirname(__file__), 'twitter_cookies.txt')
context = BrowserContext(browser=browser, config=BrowserContextConfig(cookies_file=file_path))

async def main():
	agent = Agent(
		browser_context=context,
		task=('go to https://x.com. write a new post with the text "browser-use ftw", and submit it'),
		llm=llm,
		max_actions_per_step=4,
	)
	await agent.run(max_steps=25)
	input('Press Enter to close the browser...')
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/use-cases/twitter_post_using_cookies.py)

## Web Voyager Agent

A general-purpose web navigation agent for tasks like flight booking, hotel searching, or course finding on various websites.

```python
from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig

# Set LLM based on defined environment variables
if os.getenv('OPENAI_API_KEY'):
    llm = ChatOpenAI(
        model='gpt-4o',
    )
elif os.getenv('AZURE_OPENAI_KEY') and os.getenv('AZURE_OPENAI_ENDPOINT'):
    llm = AzureChatOpenAI(
        model='gpt-4o',
        api_version='2024-10-21',
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
        api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
    )
else:
    raise ValueError('No LLM found. Please set OPENAI_API_KEY or AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT.')

browser = Browser(
    config=BrowserConfig(
        headless=False,  # This is True in production
        disable_security=True,
        new_context_config=BrowserContextConfig(
            disable_security=True,
            minimum_wait_page_load_time=1,  # 3 on prod
            maximum_wait_page_load_time=10,  # 20 on prod
            no_viewport=False,
            window_width=1280,
            window_height=1100,
        ),
    )
)

TASK = """
Find and book a hotel in Paris with suitable accommodations for a family of four (two adults and two children) offering free cancellation for the dates of February 14-21, 2025. on https://www.booking.com/
"""

async def main():
    agent = Agent(
        task=TASK,
        llm=llm,
        browser=browser,
        validate_output=True,
        enable_memory=False,
    )
    history = await agent.run(max_steps=50)
    history.save_to_file('./tmp/history.json')
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/use-cases/web_voyager_agent.py)

## Wikipedia Banana To Quantum

Navigate Wikipedia by clicking links to get from a starting page (e.g., "Banana") to a target page (e.g., "Quantum mechanics") as quickly as possible.

```python
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig

llm = ChatOpenAI(
	model='gpt-4o',
	temperature=0.0,
)

task = 'go to https://en.wikipedia.org/wiki/Banana and click on buttons on the wikipedia page to go as fast as possible from banna to Quantum mechanics'

browser = Browser(
	config=BrowserConfig(
		new_context_config=BrowserContextConfig(
			viewport_expansion=-1,
			highlight_elements=False,
		),
	),
)
agent = Agent(task=task, llm=llm, browser=browser, use_vision=False)

async def main():
	await agent.run()
```

[View full example](https://github.com/browser-use/browser-use/tree/0.1.46/examples/use-cases/wikipedia_banana_to_quantum.py)
