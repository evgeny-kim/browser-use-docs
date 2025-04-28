# 6. Custom Functions and Extensions

Browser-Use provides extensive customization options through custom functions, hooks, and extensions. This chapter explores how to extend the library's functionality to handle specialized tasks.

## The Controller System

At the core of Browser-Use's extensibility is the `Controller` class, which provides a registry for custom actions and functions.

```python
from browser_use import Controller

# Create a controller
controller = Controller()

# Use the controller with an agent
agent = Agent(
    task="Your task",
    llm=your_llm_model,
    controller=controller
)
```

The controller allows you to register custom actions that the agent can use during task execution.

## Custom Actions

Custom actions are the primary way to extend Browser-Use with new functionality. They can be registered using the `@controller.action` or `@controller.registry.action` decorator.

### Basic Custom Action

```python
from browser_use import Controller, ActionResult

controller = Controller()

@controller.action("Read a file")
def read_file(file_path: str):
    """Custom action to read a file from disk."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return ActionResult(extracted_content=content, include_in_memory=True)
    except Exception as e:
        return ActionResult(error=f"Failed to read file: {str(e)}")
```

### Actions with Parameter Models

For more complex actions, you can define Pydantic models to validate and structure the input parameters:

```python
from pydantic import BaseModel
from typing import Optional, List
from browser_use import Controller, ActionResult

class Tweet(BaseModel):
    content: str
    hashtags: Optional[List[str]] = None
    mentions: Optional[List[str]] = None

controller = Controller()

@controller.action("Create a tweet", param_model=Tweet)
def create_tweet(tweet: Tweet):
    """Custom action to create a tweet with structured data."""
    # Process the tweet
    print(f"Tweet content: {tweet.content}")
    if tweet.hashtags:
        print(f"Hashtags: {tweet.hashtags}")
    if tweet.mentions:
        print(f"Mentions: {tweet.mentions}")

    return ActionResult(extracted_content="Tweet created successfully")
```

### Browser Context Actions

Many custom actions need access to the browser context to interact with web pages:

```python
from browser_use import Controller, ActionResult
from browser_use.browser.context import BrowserContext

controller = Controller()

@controller.action("Take full page screenshot")
async def full_page_screenshot(browser: BrowserContext):
    """Takes a screenshot of the entire page, not just the visible viewport."""
    page = await browser.get_current_page()

    # Set viewport to a large size
    await page.set_viewport_size({"width": 1280, "height": 5000})

    # Take screenshot
    screenshot = await page.screenshot(full_page=True)

    # Convert to base64 for storage/display
    import base64
    base64_screenshot = base64.b64encode(screenshot).decode('utf-8')

    return ActionResult(
        extracted_content="Screenshot taken successfully",
        include_in_memory=True,
        metadata={"screenshot": base64_screenshot}
    )
```

## Action Filters

Browser-Use allows you to control when custom actions are available to the agent using domain filters and page filters. This helps reduce decision fatigue and prevents actions from being used in inappropriate contexts.

### Domain Filters

Limit actions to specific domains:

```python
# From examples/custom-functions/action_filters.py
from browser_use import Controller

controller = Controller()
registry = controller.registry

# Action will only be available on Google domains
@registry.action(
    description='Trigger disco mode',
    domains=['google.com', '*.google.com']
)
async def disco_mode(browser: BrowserContext):
    page = await browser.get_current_page()
    await page.evaluate("""() => {
        document.styleSheets[0].insertRule('@keyframes wiggle { 0% { transform: rotate(0deg); } 50% { transform: rotate(10deg); } 100% { transform: rotate(0deg); } }');

        document.querySelectorAll("*").forEach(element => {
            element.style.animation = "wiggle 0.5s infinite";
        });
    }""")
```

### Page Filters

Use custom logic to determine when an action should be available:

```python
from playwright.async_api import Page
from browser_use import Controller

controller = Controller()
registry = controller.registry

# Custom filter function
def is_login_page(page: Page) -> bool:
    return 'login' in page.url.lower() or 'signin' in page.url.lower()

# Action only available on login pages
@registry.action(
    description='Use the force, luke',
    page_filter=is_login_page
)
async def use_the_force(browser: BrowserContext):
    page = await browser.get_current_page()
    await page.evaluate("""() => {
        document.querySelector('body').innerHTML = 'These are not the droids you are looking for';
    }""")
```

## Advanced Browser Interactions

Browser-Use supports creating custom functions for advanced browser interactions like hover, drag-drop, and file uploads.

### Element Hover

```python
# From examples/custom-functions/hover_element.py
from pydantic import BaseModel
from typing import Optional
from browser_use import Controller, ActionResult, BrowserContext

class HoverAction(BaseModel):
    index: Optional[int] = None
    xpath: Optional[str] = None
    selector: Optional[str] = None

controller = Controller()

@controller.registry.action('Hover over an element', param_model=HoverAction)
async def hover_element(params: HoverAction, browser: BrowserContext):
    """Hovers over the element specified by its index, XPath, or CSS selector."""
    if params.xpath:
        element_handle = await browser.get_locate_element_by_xpath(params.xpath)
    elif params.selector:
        element_handle = await browser.get_locate_element_by_css_selector(params.selector)
    elif params.index is not None:
        session = await browser.get_session()
        state = session.cached_state
        element_node = state.selector_map[params.index]
        element_handle = await browser.get_locate_element(element_node)
    else:
        raise Exception('Either index, xpath, or selector must be provided')

    await element_handle.hover()
    return ActionResult(
        extracted_content=f"Hovered over element successfully",
        include_in_memory=True
    )
```

### File Upload

```python
# From examples/custom-functions/file_upload.py
from browser_use import Controller, ActionResult, BrowserContext

controller = Controller()

@controller.action('Upload a file to the current page')
async def upload_file(file_path: str, element_index: int, browser: BrowserContext):
    """Uploads a file to the specified file input element."""
    try:
        # Get the DOM element by index
        dom_el = await browser.get_dom_element_by_index(element_index)
        if dom_el is None:
            return ActionResult(error=f'No element found at index {element_index}')

        # Get the file upload element
        file_upload_dom_el = dom_el.get_file_upload_element()
        if file_upload_dom_el is None:
            return ActionResult(error=f'Element at index {element_index} is not a file upload')

        # Locate the element in the browser
        file_upload_el = await browser.get_locate_element(file_upload_dom_el)

        # Upload the file
        await file_upload_el.set_input_files(file_path)

        return ActionResult(
            extracted_content=f"File {file_path} uploaded successfully",
            include_in_memory=True
        )
    except Exception as e:
        return ActionResult(error=f"Failed to upload file: {str(e)}")
```

## Custom Hooks (Before/After Step)

Browser-Use provides hook mechanisms to execute custom code before or after each agent step. This is useful for logging, monitoring, and extending the agent's behavior.

### Recording Agent Activity

```python
# From examples/custom-functions/custom_hooks_before_after_step.py
import asyncio
import json
from pathlib import Path
import requests
from pyobjtojson import obj_to_json
from browser_use import Agent

# Record agent activity before each step
async def record_activity(agent_obj):
    # Capture HTML and screenshot
    website_html = await agent_obj.browser_context.get_page_html()
    website_screenshot = await agent_obj.browser_context.take_screenshot()

    # Extract history data
    history = agent_obj.state.history

    # Convert model thoughts to JSON
    model_thoughts = obj_to_json(obj=history.model_thoughts(), check_circular=False)
    model_thoughts_last = model_thoughts[-1] if model_thoughts else None

    # Get model outputs
    model_outputs = agent_obj.state.history.model_outputs()
    model_outputs_json = obj_to_json(obj=model_outputs, check_circular=False)
    model_outputs_last = model_outputs_json[-1] if model_outputs_json else None

    # Get model actions
    model_actions = agent_obj.state.history.model_actions()
    model_actions_json = obj_to_json(obj=model_actions, check_circular=False)
    model_actions_last = model_actions_json[-1] if model_actions_json else None

    # Get URLs
    urls = agent_obj.state.history.urls()
    urls_json = obj_to_json(obj=urls, check_circular=False)
    urls_last = urls_json[-1] if urls_json else None

    # Create step summary
    model_step_summary = {
        'website_html': website_html,
        'website_screenshot': website_screenshot,
        'url': urls_last,
        'model_thoughts': model_thoughts_last,
        'model_outputs': model_outputs_last,
        'model_actions': model_actions_last,
    }

    # Send data to API endpoint
    send_agent_history_step(model_step_summary)

# Function to send data to API
def send_agent_history_step(data):
    url = 'http://127.0.0.1:9000/post_agent_history_step'
    response = requests.post(url, json=data)
    return response.json()

# Create agent with hook
async def main():
    agent = Agent(
        task="Your task",
        llm=your_llm,
        # Add the hook for each step
        on_step_start=record_activity,
    )
    await agent.run()
```

### Implementing a FastAPI Server for Logging

To complement the recording hook, you can set up a FastAPI server to receive and store the logs:

```python
# From examples/custom-functions/custom_hooks_before_after_step.py
import json
from pathlib import Path
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post('/post_agent_history_step')
async def post_agent_history_step(request: Request):
    data = await request.json()

    # Ensure the "recordings" folder exists
    recordings_folder = Path('recordings')
    recordings_folder.mkdir(exist_ok=True)

    # Determine the next file number
    existing_numbers = []
    for item in recordings_folder.iterdir():
        if item.is_file() and item.suffix == '.json':
            try:
                file_num = int(item.stem)
                existing_numbers.append(file_num)
            except ValueError:
                # In case the file name isn't just a number
                pass

    # Get next file number
    next_number = max(existing_numbers) + 1 if existing_numbers else 1

    # Save the JSON data to the file
    file_path = recordings_folder / f'{next_number}.json'
    with file_path.open('w') as f:
        json.dump(data, f, indent=2)

    return {'status': 'ok', 'message': f'Saved to {file_path}'}

# Run the server
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9000)
```

## Clipboard Operations

Browser-Use can interact with the system clipboard for copying and pasting:

```python
# From examples/custom-functions/clipboard.py
import pyperclip
from browser_use import Controller, ActionResult

controller = Controller()

@controller.action("Copy text to clipboard")
def copy_to_clipboard(text: str):
    """Copy the provided text to the system clipboard."""
    pyperclip.copy(text)
    return ActionResult(extracted_content=f"Copied text to clipboard: {text[:30]}...")

@controller.action("Paste from clipboard")
def paste_from_clipboard():
    """Get the current contents of the system clipboard."""
    text = pyperclip.paste()
    return ActionResult(extracted_content=f"Clipboard contains: {text[:100]}...")
```

## Security and Notification Functions

Create custom functions for security operations and notifications:

```python
# From examples/custom-functions/notification.py
import platform
import subprocess
from browser_use import Controller, ActionResult

controller = Controller()

@controller.action("Send desktop notification")
def send_notification(title: str, message: str):
    """Send a desktop notification."""
    system = platform.system()

    try:
        if system == "Darwin":  # macOS
            subprocess.run([
                "osascript",
                "-e",
                f'display notification "{message}" with title "{title}"'
            ])
        elif system == "Linux":
            subprocess.run(["notify-send", title, message])
        elif system == "Windows":
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(title, message, duration=5)

        return ActionResult(
            extracted_content=f"Notification sent: {title} - {message}",
            include_in_memory=True
        )
    except Exception as e:
        return ActionResult(error=f"Failed to send notification: {str(e)}")
```

## Conclusion

Custom functions and extensions greatly enhance Browser-Use's capabilities, allowing it to handle specialized tasks and integrate with external systems. By leveraging the controller system, custom actions, filters, and hooks, you can adapt Browser-Use to fit virtually any web automation need.

In the next chapter, we'll explore browser configuration options in more detail.
