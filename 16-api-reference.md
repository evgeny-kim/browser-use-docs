# 16. API Reference

This chapter provides a comprehensive reference of the Browser-Use API classes, methods, and parameters.

## Core Components

### Agent

The primary class for controlling browser automation. Integrates LLM, browser control, and action execution.

```python
class Agent:
    def __init__(
        self,
        task: str,                     # Natural language instruction for the agent to execute
        llm: BaseChatModel,            # LangChain chat model for decision-making
        browser: Browser | None = None,            # Optional browser instance to use (creates one if None)
        browser_context: BrowserContext | None = None,  # Optional browser context (creates one if None)
        controller: Controller = Controller(),     # Registry of available actions for the agent
        sensitive_data: Optional[Dict[str, str]] = None,  # Secure data, referenced via placeholders
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,  # Actions to execute before first LLM call
        register_new_step_callback: Optional[Callable] = None,  # Callback after each step
        register_done_callback: Optional[Callable] = None,  # Callback when agent completes
        register_external_agent_status_raise_error_callback: Optional[Callable] = None,  # For remote control
        use_vision: bool = True,       # Whether to use visual information from browser
        use_vision_for_planner: bool = False,  # Whether to use vision for planning steps
        save_conversation_path: Optional[str] = None,  # Path to save conversation history
        save_conversation_path_encoding: Optional[str] = 'utf-8',  # Encoding for saved conversations
        max_failures: int = 3,         # Maximum consecutive failures before stopping
        retry_delay: int = 10,         # Seconds to wait between retries
        override_system_message: Optional[str] = None,  # Replace default system message
        extend_system_message: Optional[str] = None,  # Append to default system message
        max_input_tokens: int = 128000,  # Maximum tokens for LLM context
        validate_output: bool = False,  # Verify output before finishing
        message_context: Optional[str] = None,  # Additional context for LLM
        generate_gif: bool | str = False,  # Create GIF of browser interactions
        available_file_paths: Optional[list[str]] = None,  # Files agent can access
        include_attributes: list[str] = [...],  # HTML attributes to include in DOM representation
        max_actions_per_step: int = 10,  # Maximum actions per LLM call
        tool_calling_method: Optional[str] = 'auto',  # How to invoke tools ('auto', 'function_calling', 'raw')
        page_extraction_llm: Optional[BaseChatModel] = None,  # Model for page content extraction
        planner_llm: Optional[BaseChatModel] = None,  # Model for planning steps
        planner_interval: int = 1,     # Run planner every N steps
        is_planner_reasoning: bool = False,  # Show planner reasoning
        injected_agent_state: Optional[AgentState] = None,  # Pre-configured state
        context = None,                # Custom context object
        enable_memory: bool = True,    # Use memory management
        memory_interval: int = 10,     # Steps between memory consolidation
        memory_config: Optional[dict] = None,  # Memory configuration
    )

    async def run(self, max_steps: int = 100, on_step_start = None, on_step_end = None) -> AgentHistoryList:
        """Execute the task with the given maximum number of steps."""

    async def step(self, step_info: Optional[AgentStepInfo] = None) -> None:
        """Execute a single step of the agent's decision-making process."""

    async def take_step(self) -> tuple[bool, bool]:
        """Take a step and return (is_done, is_valid)."""

    def add_new_task(self, new_task: str) -> None:
        """Add a new task to the agent's context."""

    def pause(self) -> None:
        """Pause the agent's execution."""

    def resume(self) -> None:
        """Resume the agent's execution after pausing."""

    def stop(self) -> None:
        """Stop the agent's execution."""
```

### Browser

Manages browser instances and creates browser contexts.

```python
class Browser:
    def __init__(
        self,
        config: BrowserConfig | None = None,  # Browser configuration
    )

    async def new_context(self, config: BrowserContextConfig | None = None) -> BrowserContext:
        """Create a new browser context (similar to an incognito window)."""

    async def get_playwright_browser(self) -> PlaywrightBrowser:
        """Get the underlying Playwright browser instance."""

    async def close(self) -> None:
        """Close the browser and release resources."""
```

### BrowserConfig

Configuration settings for browser initialization.

```python
class BrowserConfig:
    wss_url: str | None = None          # WebSocket URL for remote browser connection
    cdp_url: str | None = None          # Chrome DevTools Protocol URL for connection
    browser_class: str = 'chromium'     # Browser type ('chromium', 'firefox', 'webkit')
    browser_binary_path: str | None = None  # Path to browser executable
    extra_browser_args: list[str] = []  # Additional browser launch arguments
    headless: bool = False              # Run without visible UI (not recommended)
    disable_security: bool = True       # Disable security features for automation
    deterministic_rendering: bool = False  # Make rendering consistent across platforms
    keep_alive: bool = False            # Keep browser open after context closes
    proxy: ProxySettings | None = None  # Proxy configuration
    new_context_config: BrowserContextConfig = BrowserContextConfig()  # Default context settings
```

### BrowserContext

Represents an isolated browser session similar to an incognito window.

```python
class BrowserContext:
    def __init__(
        self,
        browser: 'Browser',              # Parent browser instance
        config: BrowserContextConfig | None = None,  # Context configuration
        state: Optional[BrowserContextState] = None,  # Pre-configured state
    )

    async def close(self) -> None:
        """Close the browser context and release resources."""

    async def get_current_page(self) -> Page:
        """Get the currently active page/tab."""

    async def get_page_html(self) -> str:
        """Get the HTML content of the current page."""

    async def get_page_text(self) -> str:
        """Get text content from the current page."""

    async def take_screenshot(self, full_page: bool = False) -> str:
        """Capture screenshot as base64-encoded string."""

    async def remove_highlights(self) -> None:
        """Remove element highlighting from page."""

    async def navigate_to(self, url: str) -> None:
        """Navigate to a URL and wait for load."""

    async def refresh_page(self) -> None:
        """Reload the current page."""

    async def go_back(self) -> None:
        """Navigate to previous page in history."""

    async def go_forward(self) -> None:
        """Navigate to next page in history."""

    async def wait_for_element(self, selector: str, timeout: int = 5000) -> None:
        """Wait for element to appear on page."""

    async def get_state(self) -> BrowserState:
        """Get complete state of the browser for agent."""

    async def get_selector_map(self) -> dict:
        """Get map of element indices to DOM nodes."""

    async def get_dom_element_by_index(self, index: int) -> DOMElementNode:
        """Get DOM element by highlight index."""

    async def get_locate_element(self, element: DOMElementNode) -> Optional[ElementHandle]:
        """Find element handle from DOM element."""

    async def get_locate_element_by_css_selector(self, css_selector: str) -> Optional[ElementHandle]:
        """Find element by CSS selector."""

    async def get_locate_element_by_xpath(self, xpath: str) -> Optional[ElementHandle]:
        """Find element by XPath."""

    async def get_locate_element_by_text(self, text: str, nth: int = 1, element_type: str = None) -> Optional[ElementHandle]:
        """Find element by visible text content."""

    async def get_tabs_info(self) -> list[TabInfo]:
        """Get information about all open tabs."""

    async def is_file_uploader(self, element_node: DOMElementNode) -> bool:
        """Check if element is a file upload input."""

    async def switch_to_tab(self, tab_id: int) -> None:
        """Switch to specified tab by ID."""

    async def open_tab(self, url: str = "about:blank") -> None:
        """Open a new tab with specified URL."""

    async def close_tab(self, tab_id: int = None) -> None:
        """Close specified tab or current tab."""
```

### BrowserContextConfig

Configuration settings for browser contexts.

```python
class BrowserContextConfig:
    cookies_file: str | None = None     # Path to cookies file for persistence
    minimum_wait_page_load_time: float = 0.25  # Minimum seconds to wait for page load
    wait_for_network_idle_page_load_time: float = 0.5  # Wait for network to quiet down
    maximum_wait_page_load_time: float = 5  # Maximum seconds to wait for page load
    wait_between_actions: float = 0.5   # Seconds to wait between actions
    disable_security: bool = True       # Disable security features
    browser_window_size: BrowserContextWindowSize = {'width': 1280, 'height': 1100}  # Window dimensions
    no_viewport: Optional[bool] = None  # Disable viewport emulation
    save_recording_path: str | None = None  # Path to save video recording
    save_downloads_path: str | None = None  # Path to save downloaded files
    save_har_path: str | None = None    # Path to save HAR network logs
    trace_path: str | None = None       # Path to save Playwright traces
    locale: str | None = None           # Browser locale (e.g., 'en-US')
    user_agent: str = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'  # User agent string
    highlight_elements: bool = True     # Visually highlight elements
    viewport_expansion: int = 500       # Pixels to expand viewport for DOM capture
    allowed_domains: list[str] | None = None  # Restrict navigation to specific domains
    include_dynamic_attributes: bool = True  # Include dynamic attributes in selectors
    http_credentials: dict[str, str] | None = None  # HTTP Basic Auth credentials
    keep_alive: bool = False            # Keep context alive after agent finishes
    is_mobile: bool | None = None       # Emulate mobile device
    has_touch: bool | None = None       # Emulate touch screen
    geolocation: dict | None = None     # Emulate geolocation
    permissions: list[str] | None = None  # Browser permissions to grant
    timezone_id: str | None = None      # Time zone ID (e.g., 'America/New_York')
```

### Controller

Registry of functions/actions available to the agent.

```python
class Controller:
    def __init__(
        self,
        exclude_actions: list[str] = [],  # Actions to exclude from registration
        output_model: Optional[Type[BaseModel]] = None,  # Custom output model
    )

    # Decorator for registering actions
    def action(self, description: str, param_model = None, domains = None, page_filter = None):
        """Register a function as an action available to the agent."""

    # Registry of available actions
    registry: Registry
```

### Registry

Core registry for actions that can be used by agents.

```python
class Registry:
    def __init__(self, exclude_actions: list[str] | None = None):
        """Initialize registry, optionally excluding specific actions."""

    # Decorator for registering actions
    def action(self, description: str, param_model = None, domains = None, page_filter = None):
        """Register a function as an action with optional domain/page filters."""

    # Create Pydantic model for actions
    def create_action_model(self, include_actions: Optional[list[str]] = None, page = None) -> Type[ActionModel]:
        """Create a Pydantic model containing available actions."""

    # Get string description of actions for prompt
    def get_prompt_description(self, page = None) -> str:
        """Get textual description of available actions for LLM prompts."""
```

### DOM Components

Services and models for interacting with the Document Object Model.

```python
class DOMService:
    def __init__(self, page: 'Page'):
        """Initialize DOM service for a Playwright page."""

    async def get_clickable_elements(self, highlight_elements: bool = True, viewport_expansion: int = 0) -> BrowserState:
        """Get clickable elements from the page DOM."""

    async def _build_dom_tree(self, highlight_elements: bool, focus_element: int, viewport_expansion: int) -> tuple[DOMElementNode, SelectorMap]:
        """Build DOM tree representation with interactive elements."""

    async def _construct_dom_tree(self, eval_page: dict) -> tuple[DOMElementNode, SelectorMap]:
        """Construct a Python DOM tree from JavaScript evaluation."""
```

```python
@dataclass
class DOMElementNode:
    tag_name: str                   # HTML tag name (e.g., 'div', 'a')
    xpath: str                       # XPath to the element
    attributes: Dict[str, str]       # HTML attributes
    children: List[DOMBaseNode]      # Child nodes
    is_visible: bool                 # Whether element is visible
    parent: Optional['DOMElementNode']  # Parent node
    is_interactive: bool = False     # Whether element is interactive
    is_top_element: bool = False     # Whether element is visually on top
    is_in_viewport: bool = False     # Whether in current viewport
    shadow_root: bool = False        # Whether element has shadow DOM
    highlight_index: Optional[int] = None  # Index for highlighting
    viewport_coordinates: Optional[CoordinateSet] = None  # Viewport coordinates
    page_coordinates: Optional[CoordinateSet] = None  # Page coordinates
    viewport_info: Optional[ViewportInfo] = None  # Viewport information

    def get_all_text_till_next_clickable_element(self, max_depth: int = -1) -> str:
        """Get all text content up to next interactive element."""

    def clickable_elements_to_string(self, include_attributes: list[str] | None = None) -> str:
        """Convert clickable elements to string representation."""
```

### Message & Memory Components

Manages conversation history and memory for the agent.

```python
class MessageManager:
    def __init__(
        self,
        task: str,                   # The agent's task
        system_message: SystemMessage,  # System message for LLM
        settings: MessageManagerSettings = MessageManagerSettings(),  # Message settings
        state: MessageManagerState = MessageManagerState(),  # Message state
    )

    def get_messages(self) -> List[BaseMessage]:
        """Get current conversation messages for LLM."""

    def add_state_message(self, state: BrowserState, result: list[ActionResult] | None) -> None:
        """Add browser state information to conversation."""

    def add_model_output(self, output: AgentOutput) -> None:
        """Add model's output to conversation history."""

    def add_new_task(self, new_task: str) -> None:
        """Add a new task to the conversation."""

    def cut_messages(self) -> None:
        """Reduce message history to fit within token limits."""
```

```python
class Memory:
    def __init__(
        self,
        message_manager: MessageManager,  # Message manager to optimize
        llm: BaseChatModel,             # LLM for memory summarization
        settings: MemorySettings,       # Memory settings
    )

    def create_procedural_memory(self, current_step: int) -> None:
        """Create procedural memory by summarizing conversation history."""
```

## Data Models

### Action Models

Models related to agent actions and responses.

```python
class ActionResult:
    is_done: bool = False              # Whether task is complete
    extracted_content: Optional[str] = None  # Content extracted from page
    error: Optional[str] = None        # Error message if action failed
    include_in_memory: bool = False    # Whether to include in memory
```

```python
class AgentOutput:
    current_state: AgentBrain          # Agent's thought process
    action: List[ActionModel]          # Actions to execute
```

```python
class AgentBrain:
    evaluation_previous_goal: str      # Evaluation of previous step
    memory: str                        # Agent's memory of past events
    next_goal: str                     # Next goal to accomplish
```

### Agent History Models

Models for tracking agent execution history.

```python
class AgentHistoryList:
    history: List[AgentHistory]        # List of history items

    def is_done(self) -> bool:
        """Check if agent has completed task."""

    def final_result(self) -> Optional[str]:
        """Get final result of agent execution."""

    def errors(self) -> list[str | None]:
        """Get all errors from history."""

    def has_errors(self) -> bool:
        """Check if history contains errors."""

    def urls(self) -> list[str | None]:
        """Get all URLs visited."""

    def screenshots(self) -> list[str | None]:
        """Get all screenshots taken."""

    def action_names(self) -> list[str]:
        """Get names of all actions executed."""

    def model_thoughts(self) -> list[AgentBrain]:
        """Get all agent thought processes."""

    def model_outputs(self) -> list[AgentOutput]:
        """Get all model outputs."""

    def model_actions(self) -> list[dict]:
        """Get all actions with parameters."""

    def action_results(self) -> list[ActionResult]:
        """Get all action results."""

    def extracted_content(self) -> list[str]:
        """Get all extracted content."""
```

```python
class AgentHistory:
    model_output: AgentOutput          # Agent's output (thoughts and actions)
    result: List[ActionResult]         # Results of actions
    state: BrowserStateHistory         # Browser state

    def is_done(self) -> bool:
        """Check if history item marks completion."""

    def get_final_result(self) -> Optional[str]:
        """Get final result text if present."""
```

### Browser State Models

Models representing browser state.

```python
class BrowserState:
    url: str                          # Current page URL
    title: str                        # Page title
    tabs: list[TabInfo]               # Open tabs
    element_tree: DOMElementNode      # DOM tree
    selector_map: SelectorMap         # Map of element indices
    screenshot: Optional[str] = None  # Base64 encoded screenshot
    pixels_above: int = 0             # Pixels above viewport
    pixels_below: int = 0             # Pixels below viewport
    browser_errors: list[str] = []    # Browser errors
```

```python
class TabInfo:
    page_id: int                      # Tab ID
    url: str                          # Tab URL
    title: str                        # Tab title
```

## Errors

Error classes for handling exceptions.

```python
class BrowserError(Exception):
    """Base class for all browser errors"""
```

```python
class URLNotAllowedError(BrowserError):
    """Error raised when a URL is not allowed"""
```

```python
class AgentError:
    """Container for agent error handling"""

    @staticmethod
    def format_error(error: Exception, include_trace: bool = False) -> str:
        """Format error message with optional stack trace."""
```

## Default Settings

Default setting classes for various components.

```python
class AgentSettings:
    use_vision: bool                  # Use visual information
    use_vision_for_planner: bool      # Use vision for planning
    save_conversation_path: Optional[str]  # Save conversation to file
    save_conversation_path_encoding: Optional[str]  # File encoding
    max_failures: int                 # Max consecutive failures
    retry_delay: int                  # Seconds between retries
    override_system_message: Optional[str]  # Replace system message
    extend_system_message: Optional[str]  # Add to system message
    max_input_tokens: int             # Max tokens for input
    validate_output: bool             # Validate output before finishing
    message_context: Optional[str]    # Additional context
    generate_gif: bool | str          # Create GIF of execution
    available_file_paths: Optional[list[str]]  # Available files
    include_attributes: list[str]     # HTML attributes to include
    max_actions_per_step: int         # Max actions per step
    tool_calling_method: Optional[str]  # Tool calling method
    page_extraction_llm: BaseChatModel  # Model for extraction
    planner_llm: Optional[BaseChatModel]  # Model for planning
    planner_interval: int             # Steps between planning
    is_planner_reasoning: bool        # Show planner reasoning
    enable_memory: bool               # Use memory management
    memory_interval: int              # Steps between memory creation
    memory_config: Optional[dict]     # Memory configuration
```

```python
class MemorySettings:
    agent_id: str                     # Unique ID for agent
    interval: int = 10                # Steps between memory creation
    config: Optional[dict] | None = None  # Memory configuration
```

```python
class MessageManagerSettings:
    max_input_tokens: int = 128000    # Maximum input tokens
    estimated_characters_per_token: int = 3  # For token estimation
    image_tokens: int = 800           # Tokens per image
    include_attributes: list[str] = []  # HTML attributes to include
    message_context: Optional[str] = None  # Additional context
    sensitive_data: Optional[Dict[str, str]] = None  # Secure data
    available_file_paths: Optional[List[str]] = None  # Available files
```
