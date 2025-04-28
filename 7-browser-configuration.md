# 7. Browser Configuration

Browser-Use provides extensive configuration options for controlling browser behavior, appearance, and security settings. This chapter explores the various configuration options available and how to use them effectively.

## Browser Configuration Basics

The `Browser` class in Browser-Use accepts a `BrowserConfig` object that controls browser behavior at a global level:

```python
from browser_use import Browser, BrowserConfig

browser = Browser(
    config=BrowserConfig(
        # Browser configuration options
        headless=False,  # Run with visible browser window
        disable_security=False,  # Keep security features enabled
        stealth_mode=True,  # Enable anti-bot detection measures
    )
)
```

## BrowserConfig Options

The `BrowserConfig` class accepts a wide range of configuration options:

| Option                | Type                 | Default | Description                                      |
| --------------------- | -------------------- | ------- | ------------------------------------------------ |
| `headless`            | bool                 | `True`  | Run browser in headless mode (no visible window) |
| `browser_binary_path` | str                  | `None`  | Path to browser executable                       |
| `disable_security`    | bool                 | `False` | Disable browser security features                |
| `launch_args`         | List[str]            | `[]`    | Additional browser launch arguments              |
| `stealth_mode`        | bool                 | `False` | Enable anti-detection features                   |
| `cdp_url`             | str                  | `None`  | URL for Chrome DevTools Protocol                 |
| `keep_alive`          | bool                 | `False` | Keep browser open between sessions               |
| `new_context_config`  | BrowserContextConfig | `None`  | Default config for new contexts                  |

## BrowserContextConfig Options

The `BrowserContextConfig` class configures the browser context (similar to an incognito window):

```python
from browser_use import Browser, BrowserConfig, BrowserContextConfig

browser = Browser(
    config=BrowserConfig(
        headless=False,
        new_context_config=BrowserContextConfig(
            browser_window_size={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            locale="en-US",
            timezone_id="America/New_York",
        )
    )
)
```

Key options for `BrowserContextConfig` include:

| Option                        | Type      | Default              | Description                       |
| ----------------------------- | --------- | -------------------- | --------------------------------- |
| `viewport`                    | Dict      | `None`               | Viewport dimensions               |
| `browser_window_size`         | Dict      | `None`               | Browser window dimensions         |
| `user_agent`                  | str       | `None`               | Custom user agent string          |
| `locale`                      | str       | `"en-US"`            | Browser locale                    |
| `timezone_id`                 | str       | `"America/New_York"` | Browser timezone                  |
| `geolocation`                 | Dict      | `None`               | Simulated geolocation coordinates |
| `permissions`                 | List[str] | `[]`                 | Browser permissions to grant      |
| `disable_security`            | bool      | `False`              | Disable security features         |
| `minimum_wait_page_load_time` | float     | `3.0`                | Min wait time for page loads      |
| `maximum_wait_page_load_time` | float     | `20.0`               | Max wait time for page loads      |
| `trace_path`                  | str       | `None`               | Path for tracing data             |

## Stealth Mode Configuration

Stealth mode helps avoid bot detection by configuring the browser to appear more like a real user:

```python
# From examples/browser/stealth.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig

# Configure a browser with stealth settings
browser = Browser(
    config=BrowserConfig(
        headless=False,
        disable_security=False,
        stealth_mode=True,  # Enable stealth mode
        keep_alive=True,
        new_context_config=BrowserContextConfig(
            keep_alive=True,
            disable_security=False,
        ),
    )
)

# Test the browser against bot detectors
async def main():
    llm = ChatOpenAI(model='gpt-4o')

    # Test against various bot detection sites
    agent = Agent(
        task="Go to https://bot-detector.rebrowser.net/ and verify that all the bot checks are passed.",
        llm=llm,
        browser=browser,
    )
    await agent.run()

    # Test against a commercial website with bot protection
    agent = Agent(
        task="Go to https://www.webflow.com/ and verify that the page is not blocked by a bot check.",
        llm=llm,
        browser=browser,
    )
    await agent.run()
```

Stealth mode applies several optimizations:

- Modified WebDriver behavior
- Randomized fingerprinting
- Masked automation indicators
- Browser property emulation
- Consistent timezone and locale

## CDP (Chrome DevTools Protocol) Integration

Browser-Use can connect to an existing Chrome instance using the Chrome DevTools Protocol:

```python
# From examples/browser/using_cdp.py
from browser_use import Browser, BrowserConfig

# Connect to an existing Chrome instance
browser = Browser(
    config=BrowserConfig(
        headless=False,
        cdp_url='http://localhost:9222',  # Connect to running Chrome instance
    )
)
```

To use CDP:

1. Create a shortcut for Chrome with the `--remote-debugging-port=9222` argument
2. Launch Chrome using this shortcut
3. Verify CDP is running at `http://localhost:9222/json/version`
4. Configure Browser-Use to connect to this instance

## Security Settings

Browser-Use allows fine-grained control over browser security settings:

### Disabling Security Features

```python
browser = Browser(
    config=BrowserConfig(
        disable_security=True,  # Disable security at browser level
        new_context_config=BrowserContextConfig(
            disable_security=True,  # Disable security at context level
        )
    )
)
```

Disabling security can help:

- Access sites with self-signed certificates
- Bypass CORS restrictions for testing
- Handle mixed content (HTTP/HTTPS)
- Work with legacy sites that have security issues

### Caution with Security Settings

Disabling security features should be done with caution:

- Only disable security in controlled environments
- Never disable security when accessing sensitive websites
- Consider using it only for specific sites that require it
- Re-enable security after the task is complete

## Browser Launch Arguments

You can pass custom arguments directly to the browser using `launch_args`:

```python
browser = Browser(
    config=BrowserConfig(
        launch_args=[
            "--disable-gpu",  # Disable GPU acceleration
            "--no-sandbox",   # Disable sandboxing (use with caution)
            "--disable-dev-shm-usage",  # Avoid issues in containerized environments
            "--window-size=1920,1080",  # Set window size
        ]
    )
)
```

Common use cases for launch arguments:

- Performance tuning
- Memory management
- Compatibility settings
- Feature enablement/disablement

## Custom User Agent Configuration

Configure a custom user agent to change how the browser identifies itself:

```python
browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
    )
)
```

User agent customization can help:

- Test website behavior with different browsers
- Access sites that require specific browsers
- Avoid user agent-based bot detection
- Test mobile-specific content

## Viewport and Window Size Configuration

Control the browser's appearance and rendering with viewport and window size settings:

```python
browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig(
            # Viewport is the visible part of the page
            viewport={
                "width": 1280,
                "height": 800,
            },
            # Browser window size (can be larger than viewport)
            browser_window_size={
                "width": 1280,
                "height": 900,  # Extra height for browser UI
            },
        )
    )
)
```

## Geolocation and Locale Settings

Simulate different locations and language preferences:

```python
browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig(
            # Set geolocation to New York City
            geolocation={
                "latitude": 40.7128,
                "longitude": -74.0060,
                "accuracy": 100,  # Accuracy in meters
            },
            # Set language and region
            locale="en-US",
            timezone_id="America/New_York",
        )
    )
)
```

## Performance Settings

Configure page load wait times to optimize for different scenarios:

```python
browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig(
            # For fast networks or testing
            minimum_wait_page_load_time=1.0,
            maximum_wait_page_load_time=10.0,

            # For production or slow networks
            # minimum_wait_page_load_time=3.0,
            # maximum_wait_page_load_time=20.0,
        )
    )
)
```

## Tracing and Debugging

Enable tracing to capture detailed browser activity for debugging:

```python
browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig(
            # Enable tracing and save to file
            trace_path="./traces/browser_session",
        )
    )
)
```

## Real Browser vs Headless Mode

Browser-Use supports both visible (real) browsers and headless browsers:

### Real Browser Mode

```python
# From examples/browser/real_browser.py
browser = Browser(
    config=BrowserConfig(
        headless=False,  # Show the browser window
        browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    )
)
```

Benefits of real browser mode:

- Visual debugging of automation
- Better handling of some complex websites
- Easier troubleshooting
- More accurate simulation of user behavior

### Headless Mode

```python
browser = Browser(
    config=BrowserConfig(
        headless=True,  # Default: run without visible UI
    )
)
```

Benefits of headless mode:

- Lower resource usage
- Faster execution
- Better for production environments
- Works well in containerized environments

## Conclusion

Browser configuration options in Browser-Use provide extensive control over browser behavior, allowing you to fine-tune the automation experience for different use cases. By understanding and leveraging these configuration options, you can optimize browser behavior for performance, compatibility, and security.

In the next chapter, we'll explore advanced features of Browser-Use, including multi-tab handling, iframe support, and custom system prompts.
