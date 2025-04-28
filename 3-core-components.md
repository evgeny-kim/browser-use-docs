# 3. Core Components of Browser-Use

Browser-Use consists of several key components that work together to enable AI-driven browser automation. This chapter explains these core components and their interactions.

## Architecture Overview

Browser-Use implements a layered architecture:

1. **Agent Layer**: Coordinates the LLM and browser, manages task execution
2. **Browser Layer**: Handles browser automation and web interactions
3. **Context Layer**: Manages browser contexts, pages, and state

The flow of control typically follows this pattern:

- Agent receives a task and initializes the browser
- LLM analyzes the task and generates actions
- Agent translates these into browser operations
- Browser executes actions and captures results
- Results feed back to the LLM for next steps

## The Agent Class

The `Agent` class is the primary interface for Browser-Use. It coordinates the LLM, browser, and task execution.

### Key Properties

```python
class Agent:
    def __init__(
        task: str,                  # Natural language task description
        llm: Any,                   # LLM instance (e.g., ChatOpenAI)
        browser: Optional[Browser] = None,  # Browser instance (created if None)
        extend_system_message: Optional[str] = None,  # Additional instructions
        override_system_message: Optional[str] = None,  # Replace default prompt
        validate_output: bool = False,  # Enable output validation
        # Additional parameters...
    ):
        ...
```

### Agent Methods

```python
async def run(self, max_steps: Optional[int] = None) -> AgentHistory:
    """Run the agent to complete the task.

    Args:
        max_steps: Maximum number of steps before termination

    Returns:
        AgentHistory object containing task results and history
    """

async def close(self) -> None:
    """Close the agent and release resources."""
```

## The Browser Class

The `Browser` class manages browser instances and provides an interface for browser automation.

### Creating a Browser

```python
from browser_use import Browser, BrowserConfig

browser = Browser(
    config=BrowserConfig(
        headless=False,  # Run browser visibly
        browser_binary_path='/path/to/chrome',  # Optional Chrome path
        disable_security=True,  # Disable security features (CORS, etc.)
        new_context_config=BrowserContextConfig(...)  # Config for new contexts
    )
)
```

### Browser Methods

```python
async def new_context(self, config: Optional[BrowserContextConfig] = None) -> BrowserContext:
    """Create a new browser context (similar to incognito window)."""

async def close(self) -> None:
    """Close the browser and all active contexts."""
```

## BrowserConfig

The `BrowserConfig` class configures browser behavior:

```python
class BrowserConfig:
    headless: bool = True  # Run in headless mode
    browser_binary_path: Optional[str] = None  # Path to browser executable
    disable_security: bool = False  # Disable security restrictions
    launch_args: List[str] = []  # Browser launch arguments
    new_context_config: Optional[BrowserContextConfig] = None  # Default context config
    stealth_mode: bool = False  # Enable anti-detection features
    cdp_url: Optional[str] = None  # URL for Chrome DevTools Protocol
```

## BrowserContext

A `BrowserContext` represents an isolated browser session (similar to an incognito window). Each context can contain multiple pages/tabs.

### Creating a BrowserContext

```python
browser_context = await browser.new_context(
    config=BrowserContextConfig(...)
)
```

### BrowserContext Methods

```python
async def new_page(self) -> Page:
    """Create a new page (tab) in this context."""

async def get_page_html(self) -> str:
    """Get HTML content of the current page."""

async def get_page_text(self) -> str:
    """Get text content of the current page."""

async def take_screenshot(self) -> str:
    """Take screenshot of current page."""

async def close(self) -> None:
    """Close this context and all its pages."""
```

## BrowserContextConfig

The `BrowserContextConfig` class configures browser context behavior:

```python
class BrowserContextConfig:
    viewport: Optional[Dict[str, int]] = None  # Viewport dimensions
    user_agent: Optional[str] = None  # Custom user agent
    locale: str = "en-US"  # Browser locale
    timezone_id: str = "America/New_York"  # Browser timezone
    geolocation: Optional[Dict[str, float]] = None  # Geolocation coordinates
    permissions: List[str] = []  # Browser permissions
    disable_security: bool = False  # Disable security features
    browser_window_size: Optional[Dict[str, int]] = None  # Window dimensions
    minimum_wait_page_load_time: float = 3.0  # Min wait time for page loads
    maximum_wait_page_load_time: float = 20.0  # Max wait time for page loads
    trace_path: Optional[str] = None  # Path for tracing data
```

## Practical Example

Here's a complete example showing how these components work together:

```python
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig

load_dotenv()

# Create a browser configuration
browser_config = BrowserConfig(
    headless=False,
    browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    new_context_config=BrowserContextConfig(
        browser_window_size={"width": 1280, "height": 800},
        minimum_wait_page_load_time=1.0,
        maximum_wait_page_load_time=10.0
    )
)

# Initialize the browser
browser = Browser(config=browser_config)

async def main():
    # Create the agent
    agent = Agent(
        task='In docs.google.com write my Papa a quick letter',
        llm=ChatOpenAI(model='gpt-4o'),
        browser=browser,
    )

    # Run the agent
    result = await agent.run()

    # Close resources when done
    await browser.close()

    # Show the result
    print(result.final_answer)

if __name__ == '__main__':
    asyncio.run(main())
```

## Component Interaction

Here's how the components interact during a typical Browser-Use session:

1. The `Agent` initializes with a task and LLM
2. If no `Browser` is provided, the `Agent` creates one
3. The `Browser` creates a `BrowserContext` using `BrowserContextConfig`
4. The `Agent` sends the task to the LLM, which generates actions
5. The `Agent` translates LLM outputs into `BrowserContext` method calls
6. The `BrowserContext` executes these actions in the browser
7. Results are captured and sent back to the LLM
8. This cycle repeats until the task is complete or max_steps is reached

Understanding these core components provides the foundation for working with Browser-Use effectively and extending its functionality for custom use cases.
