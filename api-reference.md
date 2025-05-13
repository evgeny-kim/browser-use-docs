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
        tool_calling_method: ToolCallingMethod | None = 'auto',  # How to invoke tools ('auto', 'function_calling', 'raw', 'tools'). ToolCallingMethod is a Literal type.
        page_extraction_llm: Optional[BaseChatModel] = None,  # Model for page content extraction
        planner_llm: Optional[BaseChatModel] = None,  # Model for planning steps
        planner_interval: int = 1,     # Run planner every N steps
        is_planner_reasoning: bool = False,  # Show planner reasoning
        extend_planner_system_message: str | None = None, # Append to default planner system message
        injected_agent_state: Optional[AgentState] = None,  # Pre-configured state (AgentState will be defined later)
        context = None,                # Custom context object
        enable_memory: bool = True,    # Use memory management
        memory_config: MemoryConfig | None = None,  # Memory configuration (MemoryConfig is defined in Data Models/Configuration Models)
        save_playwright_script_path: str | None = None, # Path to save a Playwright script of the session
        source: str | None = None      # Source of the agent run, for telemetry
    )

    async def run(self, max_steps: int = 100, on_step_start: Optional[Callable[['Agent'], Awaitable[None]]] = None, on_step_end: Optional[Callable[['Agent'], Awaitable[None]]] = None) -> AgentHistoryList:
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

    async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
        """Get next action from LLM based on current state."""

    async def multi_act(self, actions: list[ActionModel], check_for_new_elements: bool = True) -> list[ActionResult]:
        """Execute multiple actions."""

    async def rerun_history(self, history: AgentHistoryList, max_retries: int = 3, skip_failures: bool = True, delay_between_actions: float = 2.0) -> list[ActionResult]:
        """Rerun a previously recorded agent history."""

    async def load_and_rerun(self, history_file: str | Path | None = None, **kwargs) -> list[ActionResult]:
        """Load an agent history from a file and rerun it."""

    def save_history(self, file_path: str | Path | None = None) -> None:
        """Save the current agent history to a file."""

    async def log_completion(self) -> None:
        """Log the completion of the task."""

    async def close(self) -> None:
        """Close all resources used by the agent, including the browser."""

    @property
    def message_manager(self) -> MessageManager: # MessageManager will be defined/referenced later
        """Access the agent's message manager."""
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
    chrome_remote_debugging_port: int | None = 9222 # Optional port for Chrome DevTools Protocol, defaults to 9222
    extra_browser_args: list[str] = []  # Additional browser launch arguments
    headless: bool = False              # Run without visible UI (not recommended)
    disable_security: bool = False      # Disable security features (e.g., CORS, CSP). Setting to True is dangerous and should be used with caution.
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
        state: Optional[BrowserContextState] = None,  # Pre-configured state (BrowserContextState will be defined later)
    )

    async def close(self) -> None:
        """Close the browser context and release resources."""

    async def get_current_page(self) -> Page:
        """Get the currently active page/tab that the agent is interacting with."""

    async def get_agent_current_page(self) -> Page:
        """Get the page that the agent is currently working with, ensuring recovery if the tab reference is invalid."""

    async def get_page_html(self) -> str:
        """Get the HTML content of the current page."""

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

    async def wait_for_element(self, selector: str, timeout: float = 5000.0) -> None:
        """Wait for element to appear on page."""

    async def get_state(self, cache_clickable_elements_hashes: bool = False) -> BrowserState:
        """Get complete state of the browser for agent.
           cache_clickable_elements_hashes: If True, cache hashes of clickable elements to identify new elements in subsequent states.
        """

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

    async def get_locate_element_by_text(self, text: str, nth: Optional[int] = 0, element_type: str = None) -> Optional[ElementHandle]:
        """Find element by visible text content."""

    async def get_tabs_info(self) -> list[TabInfo]:
        """Get information about all open tabs."""

    async def is_file_uploader(self, element_node: DOMElementNode, max_depth: int = 3, current_depth: int = 0) -> bool:
        """Check if element is a file upload input."""

    async def switch_to_tab(self, page_id: int) -> None:
        """Switch to specified tab by ID."""

    async def create_new_tab(self, url: str = "about:blank") -> None:
        """Open a new tab with specified URL."""

    async def close_current_tab(self) -> None:
        """Close the agent's current tab."""

    async def execute_javascript(self, script: str) -> Any:
        """Execute JavaScript code on the agent's current page."""

    async def get_page_structure(self) -> str:
        """Get a debug view of the page structure including iframes."""

    async def get_element_by_index(self, index: int) -> Optional[ElementHandle]:
        """Get a Playwright ElementHandle by its highlight index."""

    async def save_cookies(self) -> None:
        """Save current browser context cookies to the file specified in BrowserContextConfig."""

    async def get_scroll_info(self, page: Page) -> tuple[int, int]:
        """Get the number of pixels scrollable above and below the current viewport on the given page."""

    async def reset_context(self) -> None:
        """Resets the browser context, clearing cookies, storage, and re-initializing the session."""
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
    window_width: int = 1280            # Default browser window width
    window_height: int = 1100           # Default browser window height
    no_viewport: bool = True  # When True (default for headful), browser window size determines viewport. If False, a fixed viewport (width/height above) is enforced.
    save_recording_path: str | None = None  # Path to save video recording
    save_downloads_path: str | None = None  # Path to save downloaded files
    save_har_path: str | None = None    # Path to save HAR network logs
    trace_path: str | None = None       # Path to save Playwright traces
    locale: str | None = None           # Browser locale (e.g., 'en-US')
    user_agent: str | None = None       # User agent string. If None, Playwright's default is used.
    highlight_elements: bool = True     # Visually highlight elements
    viewport_expansion: int = 0       # Pixels to expand viewport for DOM capture. 0 means only visible elements.
    allowed_domains: list[str] | None = None  # Restrict navigation to specific domains
    include_dynamic_attributes: bool = True  # Include dynamic attributes in selectors
    http_credentials: dict[str, str] | None = None  # HTTP Basic Auth credentials
    keep_alive: bool = False            # Keep context alive after agent finishes
    is_mobile: bool | None = None       # Emulate mobile device
    has_touch: bool | None = None       # Emulate touch screen
    geolocation: dict | None = None     # Emulate geolocation
    permissions: list[str] = ['clipboard-read', 'clipboard-write']  # Browser permissions to grant
    timezone_id: str | None = None      # Time zone ID (e.g., 'America/New_York')
    force_new_context: bool = False     # Forces a new browser context, even if one could be reused (e.g. with CDP).
```

### Controller

Registry of functions/actions available to the agent. The controller utilizes a `Registry` instance (accessible via the `registry` attribute) for managing actions.

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

Core registry for actions that can be used by agents. Typically accessed via `Controller.registry`.

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

### SystemPrompt

Provides the system message for the Agent, used to instruct the LLM on its role, capabilities, and constraints.

```python
class SystemPrompt:
    def __init__(
        self,
        action_description: str,                     # String describing all available actions
        max_actions_per_step: int = 10,          # Maximum number of actions the agent can take in a single step
        override_system_message: Optional[str] = None,  # Completely replace the default system message with this string
        extend_system_message: Optional[str] = None     # Append this string to the default system message
    ):
        """Initializes the SystemPrompt.

        Args:
            action_description: A string detailing the actions available to the agent.
            max_actions_per_step: The maximum number of actions the agent can perform in one step.
            override_system_message: If provided, this string will be used as the entire system message, ignoring the default template.
            extend_system_message: If provided, this string will be appended to the generated system message.
        """

    def get_system_message(self) -> SystemMessage:
        """Get the SystemMessage object to be used by the LLM.

        Returns:
            SystemMessage: The LangChain SystemMessage object containing the formatted prompt.
        """
```

### DOM Components

Services and models for interacting with the Document Object Model.

```python
class DOMService:
    def __init__(self, page: 'Page'):
        """Initialize DOM service for a Playwright page."""

    async def get_clickable_elements(self, highlight_elements: bool = True, focus_element: int = -1, viewport_expansion: int = 0) -> DOMState:
        """Get clickable elements from the page DOM. Returns a DOMState object."""

    async def get_cross_origin_iframes(self) -> list[str]:
        """Get a list of URLs for cross-origin iframes within the current page."""

    async def _build_dom_tree(self, highlight_elements: bool, focus_element: int, viewport_expansion: int) -> tuple[DOMElementNode, SelectorMap]:
        """Build DOM tree representation with interactive elements."""

    async def _construct_dom_tree(self, eval_page: dict) -> tuple[DOMElementNode, SelectorMap]:
        """Construct a Python DOM tree from JavaScript evaluation."""
```

```python
@dataclass
class CoordinateSet:
    x: int                            # X-coordinate of the top-left corner
    y: int                            # Y-coordinate of the top-left corner
    width: int                        # Width of the area
    height: int                       # Height of the area
```

```python
@dataclass
class ViewportInfo:
    width: int                        # Width of the viewport
    height: int                       # Height of the viewport
```

```python
@dataclass
class DOMElementNode:
    tag_name: str                   # HTML tag name (e.g., 'div', 'a')
    xpath: str                       # XPath to the element
    attributes: Dict[str, str]       # HTML attributes
    children: List[DOMBaseNode]      # Child nodes (DOMBaseNode is a base for DOMElementNode and DOMTextNode)
    is_visible: bool                 # Whether element is visible
    parent: Optional['DOMElementNode']  # Parent node
    is_interactive: bool = False     # Whether element is interactive
    is_top_element: bool = False     # Whether element is visually on top
    is_in_viewport: bool = False     # Whether in current viewport
    shadow_root: bool = False        # Whether element has shadow DOM
    highlight_index: Optional[int] = None  # Index for highlighting
    viewport_coordinates: Optional[CoordinateSet] = None  # Viewport coordinates
    page_coordinates: Optional[CoordinateSet] = None  # Page coordinates
    viewport_info: Optional[ViewportInfo] = None  # Viewport information for the element's frame/document
    is_new: bool | None = None       # Whether this element is new since the last state update (useful for agent memory)

    def get_all_text_till_next_clickable_element(self, max_depth: int = -1) -> str:
        """Get all text content up to next interactive element."""

    def clickable_elements_to_string(self, include_attributes: list[str] | None = None) -> str:
        """Convert clickable elements to string representation."""

    def get_file_upload_element(self, check_siblings: bool = True) -> Optional['DOMElementNode']:
        """Checks if the current element or its children (or siblings if check_siblings is True) is an input element of type 'file'."""
```

```python
# SelectorMap is a type alias for Dict[int, DOMElementNode]
# It maps a highlight_index to its corresponding DOMElementNode.
SelectorMap = Dict[int, DOMElementNode]
```

```python
@dataclass
class DOMState:
    element_tree: DOMElementNode      # The root of the DOM tree representation
    selector_map: SelectorMap         # A map from highlight_index to DOMElementNode
```

### Message & Memory Components

Manages conversation history and memory for the agent. The `Agent` exposes a `message_manager` property. Memory is configured via `MemoryConfig` passed to the `Agent`.

#### MessageManager

The `MessageManager` class is responsible for handling the flow of messages to and from the language model, including managing context length and incorporating state information. An instance of this class is accessible via `Agent.message_manager`.

```python
class MessageManager:
    def __init__(
        self,
        task: str,                               # The agent's task
        system_message: SystemMessage,           # System message for LLM
        settings: MessageManagerSettings = MessageManagerSettings(),  # Configuration for the message manager (defined in Default Settings)
        state: MessageManagerState = MessageManagerState(),        # Initial state for the message manager (MessageManagerState is defined in Data Models)
    )

    def add_new_task(self, new_task: str) -> None:
        """Add a new task to the conversation, informing the LLM of the change."""

    def add_state_message(
        self,
        state: BrowserState,                     # Current browser state
        result: list[ActionResult] | None = None,  # Result from the last action(s)
        step_info: Optional[AgentStepInfo] = None, # Information about the current step (AgentStepInfo is defined in Data Models)
        use_vision: bool = True                  # Whether to include visual (screenshot) information in the message
    ) -> None:
        """Add browser state information to the conversation history."""

    def add_model_output(self, model_output: AgentOutput) -> None:
        """Add the language model's output (thoughts and actions) to conversation history."""

    def add_plan(self, plan: str | None, position: int | None = None) -> None:
        """Adds a plan (typically from a planner agent) into the message history at the specified position."""

    def get_messages(self) -> List[BaseMessage]:
        """Get the current list of messages prepared for the LLM, after context management."""

    def cut_messages(self) -> None:
        """Reduce message history to fit within token limits, typically by truncating the last state message or removing images."""

    def add_tool_message(self, content: str, message_type: str | None = None, **additional_kwargs: Any) -> None:
        """Adds a ToolMessage to the history, usually representing the output of a tool call."""
```

#### Memory

The `Memory` class handles the creation and management of procedural memory for the agent, helping to consolidate past interactions into a summarized form. This is an internal component of the `Agent` when memory is enabled. Its behavior is configured through the `MemoryConfig` object.

```python
class Memory:
    def __init__(
        self,
        message_manager: MessageManager,    # The MessageManager instance to work with
        llm: BaseChatModel,                 # The language model used for summarization
        config: MemoryConfig | None = None, # Configuration for memory (MemoryConfig is defined in Data Models/Configuration Models)
    )

    def create_procedural_memory(self, current_step: int) -> None:
        """Consolidates messages from the MessageManager's history to create or update procedural memory,
           replacing older messages with a summarized version.
        """
```

## Data Models

### Action Models

Models related to agent actions and responses.

```python
class ActionResult:
    is_done: bool | None = False       # Whether task is complete
    success: bool | None = None        # Whether the 'done' action indicated overall task success (True if successful, False if not, None if not a 'done' action or not applicable)
    extracted_content: Optional[str] = None  # Content extracted from page
    error: Optional[str] = None        # Error message if action failed
    include_in_memory: bool = False    # Whether to include in memory
```

```python
class AgentOutput:
    current_state: AgentBrain          # Agent's thought process
    action: List[ActionModel]          # Actions to execute. ActionModel is a base type; specific actions (e.g., click_element_by_index) will have their own parameter models.
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
        """Check if agent has completed task based on the last action result."""

    def is_successful(self) -> bool | None:
        """Check if the agent completed the task successfully. Returns True if successful, False if not, None if not done yet."""

    def final_result(self) -> Optional[str]:
        """Get final result text from the last action if the task is done."""

    def errors(self) -> list[str | None]:
        """Get all error messages from history, with None for steps without errors."""

    def has_errors(self) -> bool:
        """Check if history contains any errors."""

    def urls(self) -> list[str | None]:
        """Get all URLs visited during the agent's execution."""

    def screenshots(self) -> list[str | None]:
        """Get all base64 encoded screenshots taken."""

    def action_names(self) -> list[str]:
        """Get names of all actions executed."""

    def model_thoughts(self) -> list[AgentBrain]:
        """Get all agent thought processes (AgentBrain instances)."""

    def model_outputs(self) -> list[AgentOutput]:
        """Get all raw model outputs (AgentOutput instances)."""

    def model_actions(self) -> list[dict]:
        """Get all actions executed with their parameters."""

    def action_results(self) -> list[ActionResult]:
        """Get all action results."""

    def extracted_content(self) -> list[str]:
        """Get all content extracted by actions."""

    def total_duration_seconds(self) -> float:
        """Get total duration of all steps in seconds."""

    def total_input_tokens(self) -> int:
        """Get total approximate input tokens used across all steps."""

    def input_token_usage(self) -> list[int]:
        """Get approximate input token usage for each step."""

    def save_to_file(self, filepath: str | Path) -> None:
        """Save the agent history to a JSON file."""

    def save_as_playwright_script(
        self,
        output_path: str | Path,
        sensitive_data_keys: Optional[list[str]] = None,
        browser_config: Optional[BrowserConfig] = None,
        context_config: Optional[BrowserContextConfig] = None,
    ) -> None:
        """Generate and save a Playwright script based on the agent's history."""

    @classmethod
    def load_from_file(cls, filepath: str | Path, output_model: Type[AgentOutput]) -> 'AgentHistoryList':
        """Load an agent history from a JSON file."""

    def last_action(self) -> Optional[dict]:
        """Get the last action executed in the history, if any."""

    def model_actions_filtered(self, include: Optional[list[str]] = None) -> list[dict]:
        """Get model actions from history, filtered by a list of action names."""

    def number_of_steps(self) -> int:
        """Get the number of steps recorded in the history."""
```

```python
class AgentHistory:
    model_output: Optional[AgentOutput]  # Agent's output (thoughts and actions)
    result: List[ActionResult]         # Results of actions
    state: BrowserStateHistory         # Browser state at the time of the step (BrowserStateHistory is an internal representation)
    metadata: Optional[StepMetadata] = None # Metadata for the step, like timing and token count (StepMetadata defined below)
```

### Browser State Models

Models representing browser state.

```python
class BrowserState:
    url: str                          # Current page URL
    title: str                        # Page title
    tabs: list[TabInfo]               # Open tabs
    element_tree: DOMElementNode      # DOM tree (DOMElementNode defined in DOM Components)
    selector_map: SelectorMap         # Map of element indices
    screenshot: Optional[str] = None  # Base64 encoded screenshot
    pixels_above: int = 0             # Pixels scrollable above the current viewport
    pixels_below: int = 0             # Pixels scrollable below the current viewport
    browser_errors: list[str] = []    # Browser console errors
```

```python
class TabInfo:
    page_id: int                      # Tab ID
    url: str                          # Tab URL
    title: str                        # Tab title
    parent_page_id: Optional[int] = None # Optional ID of the parent page if this tab is a popup or iframe
```

### Other Data Models

This section includes other Pydantic models that are part of the API, often used as parameters or return types for various components.

```python
@dataclass
class AgentState:
    """Holds all state information for an Agent. Can be used for `injected_agent_state` in Agent `__init__`."""
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    n_steps: int = 1
    consecutive_failures: int = 0
    last_result: Optional[List[ActionResult]] = None
    history: AgentHistoryList = Field(default_factory=AgentHistoryList) # type: ignore
    last_plan: Optional[str] = None
    paused: bool = False
    stopped: bool = False
    message_manager_state: MessageManagerState = Field(default_factory=MessageManagerState) # type: ignore # MessageManagerState defined with MessageManager
```

```python
@dataclass
class AgentStepInfo:
    """Information about the current step, passed to `Agent.step()`."""
    step_number: int
    max_steps: int

    def is_last_step(self) -> bool:
        """Check if this is the last step."""
```

```python
class StepMetadata(BaseModel):
    """Metadata for a single agent step, found in `AgentHistory.metadata`."""
    step_start_time: float
    step_end_time: float
    input_tokens: int  # Approximate tokens from message manager for this step
    step_number: int

    @property
    def duration_seconds(self) -> float:
        """Calculate step duration in seconds."""
```

```python
class MemoryConfig(BaseModel):
    """Configuration for procedural memory, used in `Agent.__init__`."""
    agent_id: str = Field(default='browser_use_agent', min_length=1)
    memory_interval: int = Field(default=10, gt=1, lt=100) # Interval in steps for creating memory summaries
    embedder_provider: Literal['openai', 'gemini', 'ollama', 'huggingface'] = 'huggingface'
    embedder_model: str = Field(min_length=2, default='all-MiniLM-L6-v2')
    embedder_dims: int = Field(default=384, gt=10, lt=10000)
    llm_provider: Literal['langchain'] = 'langchain'
    llm_instance: Optional[BaseChatModel] = None # The LLM instance to use for summarization
    vector_store_provider: Literal['faiss'] = 'faiss'
    vector_store_base_path: str = Field(default='/tmp/mem0') # Base path for vector store persistence
```

```python
@dataclass
class BrowserContextState:
    """State of the browser context, can be used for `state` in `BrowserContext.__init__`."""
    target_id: Optional[str] = None  # CDP target ID
```

```python
class ProxySettings(BaseModel):
    """Proxy server configuration, used in `BrowserConfig`."""
    server: str                         # Proxy server URL (e.g., "http://myproxy.com:3128" or "socks5://myproxy.com:3128")
    bypass: Optional[str] = None        # Comma-separated list of hosts to bypass proxy (e.g., "localhost,*.example.com")
    username: Optional[str] = None
    password: Optional[str] = None
```

## Errors

Error classes for handling exceptions, and related utilities.

```python
class BrowserError(Exception):
    """Base class for all browser errors"""
```

```python
class URLNotAllowedError(BrowserError):
    """Error raised when a URL is not allowed"""
```

```python
class LLMException(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f'Error {status_code}: {message}')
    """Error raised for issues related to Language Model interactions."""
```

```python
class AgentError:
    """Utility class for agent error handling. Not an exception itself."""

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
    tool_calling_method: ToolCallingMethod | None = 'auto'  # How to invoke tools ('auto', 'function_calling', 'raw', 'tools'). ToolCallingMethod is a Literal type.
    page_extraction_llm: BaseChatModel  # Model for extraction
    planner_llm: Optional[BaseChatModel]  # Model for planning
    planner_interval: int             # Steps between planning
    is_planner_reasoning: bool        # Show planner reasoning
    extend_planner_system_message: str | None = None # Append to default planner system message
    save_playwright_script_path: str | None = None # Path to save a Playwright script of the session
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
