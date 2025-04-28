# 14. Performance and Optimization

Browser automation with AI can be resource-intensive, involving browser processes, language model inference, and complex DOM manipulation. This chapter explores techniques to optimize Browser-Use for performance, reliability, and efficiency.

## Browser Configuration Optimization

### Headless Mode

For production or batch processing, use headless mode to minimize resource usage:

```python
from browser_use import Browser, BrowserConfig

browser = Browser(
    config=BrowserConfig(
        headless=True,  # Run without visible UI (default)
    )
)
```

Benefits of headless mode:

- Reduced memory usage
- Lower CPU consumption
- Faster execution time
- Better scalability for parallel agents

### Page Load Time Settings

Optimize page load wait times based on your use case:

```python
browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig(
            # For fast networks or testing
            minimum_wait_page_load_time=1.0,      # Minimum wait before capturing page state
            maximum_wait_page_load_time=10.0,     # Maximum wait before proceeding
            wait_for_network_idle_page_load_time=1.0,  # Wait for network activity to cease
        )
    )
)
```

Guidelines for optimizing wait times:

- **Fast networks/simple sites**: Use shorter wait times (min: 0.5s, max: 5s)
- **Complex sites/slower networks**: Increase wait times (min: 3s, max: 20s)
- **Specific site optimization**: Test different values to find the optimal balance

### Viewport and Window Size

Configure viewport and window size for optimal rendering and interaction:

```python
browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig(
            # Viewport (visible area)
            viewport={
                "width": 1280,
                "height": 800,
            },
            # Browser window size
            browser_window_size={
                "width": 1280,
                "height": 900,  # Extra height for browser UI
            },
            # Virtual viewport expansion (for scrolling)
            viewport_expansion=500,  # Extends visible area for DOM processing
        )
    )
)
```

Optimization guidelines:

- Use larger viewport for sites with responsive layouts
- Set viewport_expansion higher for infinite scrolling pages
- Balance size with memory usage—larger viewports consume more memory

### Chrome Performance Arguments

Use specific Chrome arguments to optimize browser performance:

```python
browser = Browser(
    config=BrowserConfig(
        launch_args=[
            "--disable-gpu",  # Use CPU rendering (more consistent)
            "--disable-dev-shm-usage",  # Avoid issues in containerized environments
            "--disable-extensions",  # Disable unnecessary extensions
            "--disable-background-networking",  # Reduce background activity
            "--disable-background-timer-throttling",  # Prevent timers being throttled
            "--disable-backgrounding-occluded-windows",  # Prevent background throttling
            "--disable-breakpad",  # Disable crash reporting
            "--disable-component-extensions-with-background-pages",  # Reduce memory usage
            "--no-sandbox",  # Disable sandboxing (use with caution, security implications)
        ]
    )
)
```

> **Note**: Some arguments like `--no-sandbox` reduce security. Only use these in controlled environments.

## Model Selection Optimization

Different LLM models offer different trade-offs between performance, capability, and cost:

### Model Capability vs. Resource Usage

| Model Type            | Capabilities                                       | Resource Requirements      | Best For                                          |
| --------------------- | -------------------------------------------------- | -------------------------- | ------------------------------------------------- |
| GPT-4o                | Excellent visual understanding, reliable reasoning | High API cost              | Complex tasks, production use cases               |
| Claude 3.7            | Long context window, strong reasoning              | Medium-high API cost       | Tasks with long content                           |
| Gemini                | Good visual reasoning                              | Medium API cost            | General automation tasks                          |
| Local models (Ollama) | Basic automation                                   | No API cost, local compute | Simple tasks, development, privacy-sensitive apps |

### Model-Specific Optimization

Adjust agent parameters based on your chosen model:

```python
# For advanced models like GPT-4o (fewer steps, more actions per step)
agent = Agent(
    task="Your complex task",
    llm=ChatOpenAI(model="gpt-4o"),
    max_steps=10,  # Standard number of steps is often sufficient
    max_actions_per_step=10,  # Allow more actions per reasoning step
)

# For less capable models (more steps, fewer actions per step)
agent = Agent(
    task="Your task, broken down into simpler steps",
    llm=ChatOllama(model="llama3"),
    max_steps=20,  # Allow more steps
    max_actions_per_step=2,  # Limit actions per step
    validate_output=True,  # Add validation to ensure quality
)
```

### Specialist Models for Different Tasks

Use specialized models for different parts of the task:

```python
from langchain_openai import ChatOpenAI

# Main model for reasoning and decision-making
main_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# Smaller, faster model for page extraction
extraction_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# Create agent with specialized models
agent = Agent(
    task="Your task",
    llm=main_llm,
    page_extraction_llm=extraction_llm,  # Use smaller model for HTML processing
)
```

## Parallel Processing and Multi-Agent Systems

For complex tasks or batch processing, use parallel execution:

### Parallel Agent Execution

```python
# From examples/features/parallel_agents.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig

# Configure a shared browser
browser = Browser(
    config=BrowserConfig(
        headless=True,  # Use headless mode for parallel execution
    )
)
llm = ChatOpenAI(model='gpt-4o')

async def main():
    # Define multiple agents with different tasks
    agents = [
        Agent(task=task, llm=llm, browser=browser)
        for task in [
            'Search Google for weather in Tokyo',
            'Check Reddit front page title',
            'Look up Bitcoin price on Coinbase',
            'Find NASA image of the day',
        ]
    ]

    # Run all agents concurrently
    await asyncio.gather(*[agent.run() for agent in agents])

    # Close browser when done
    await browser.close()

asyncio.run(main())
```

### Resource Management for Parallel Agents

To prevent resource exhaustion with parallel agents:

```python
import asyncio
import resource
from browser_use import Agent, Browser, BrowserConfig

# Set resource limits (UNIX-like systems)
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

# Limit concurrency with a semaphore
MAX_CONCURRENT = 5  # Adjust based on system capabilities
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

async def run_with_semaphore(agent):
    async with semaphore:
        return await agent.run()

async def main():
    browser = Browser(config=BrowserConfig(headless=True))

    # Create 20 agents but only run 5 at a time
    agents = [
        Agent(task=f"Task {i}", llm=llm, browser=browser)
        for i in range(20)
    ]

    # Run with controlled concurrency
    results = await asyncio.gather(*[run_with_semaphore(agent) for agent in agents])

    await browser.close()
```

## Memory and DOM Optimization

Browser-Use processes DOM trees which can be memory-intensive for complex pages:

### DOM Tree Processing Optimization

Control DOM processing with viewport expansion settings:

```python
browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig(
            viewport_expansion=0,  # Limit to visible viewport only (default)
            # viewport_expansion=500,  # Expand processing area below viewport
            # viewport_expansion=2000,  # Large expansion for infinite scrolling pages
        )
    )
)
```

### Attribute Filtering

Limit which attributes are processed to reduce memory usage:

```python
agent = Agent(
    task="Your task",
    llm=llm,
    include_attributes=[
        'title',
        'type',
        'name',
        'role',
        'aria-label',
        'placeholder',
        'value',
        # Remove unnecessary attributes to improve performance
    ],
)
```

### Domain Filtering

Restrict browsing to specific domains to prevent unwanted navigation:

```python
browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig(
            allowed_domains=['google.com', 'wikipedia.org'],  # Only allow these domains
        )
    )
)
```

## Error Handling and Reliability

Implement robust error handling for production reliability:

### Rate Limit Handling

```python
import asyncio
import time
from openai import RateLimitError

async def run_with_rate_limit_handling(agent):
    max_retries = 5
    retry_count = 0
    base_delay = 2  # seconds

    while retry_count < max_retries:
        try:
            return await agent.run()
        except RateLimitError:
            retry_count += 1
            if retry_count >= max_retries:
                raise

            # Exponential backoff
            delay = base_delay * (2 ** (retry_count - 1))
            print(f"Rate limit reached. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
```

### Task Recovery

Implement checkpointing to recover from failures:

```python
import json
import os

async def run_with_checkpointing(agent, checkpoint_file='checkpoint.json'):
    # Check if checkpoint exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        # Restore agent state from checkpoint
        agent.state = checkpoint['state']
        step = checkpoint['step']
        print(f"Resuming from step {step}")
    else:
        step = 0

    try:
        # Run with periodic checkpointing
        while not agent.state.is_done():
            # Run a single step
            await agent.step()
            step += 1

            # Save checkpoint every 5 steps
            if step % 5 == 0:
                checkpoint = {
                    'state': agent.state,
                    'step': step
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f)

        return agent.state.result
    except Exception as e:
        print(f"Error: {e}. Checkpoint saved at step {step}")
        raise
```

## Performance Monitoring

Monitor and debug performance issues:

### Enabling Debug Mode

```python
from browser_use import Browser, BrowserConfig, BrowserContextConfig

browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig(
            debug_mode=True,  # Enable detailed performance logging
        )
    )
)
```

### Performance Metrics Collection

```python
import time
import asyncio
import psutil

async def measure_performance(task_fn):
    process = psutil.Process()

    # Memory before
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Time the execution
    start_time = time.time()
    result = await task_fn()
    execution_time = time.time() - start_time

    # Memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Memory usage: {mem_after - mem_before:.2f} MB")

    return result

# Usage
async def run_browser_task():
    agent = Agent(task="Your task", llm=llm)
    return await agent.run()

result = await measure_performance(run_browser_task)
```

## Model Performance Comparison Benchmarks

This table provides a comparative overview of different models' performance across key metrics:

| Model             | Success Rate | Avg. Time per Task | API Cost per Task | Visual Understanding | Memory Usage |
| ----------------- | ------------ | ------------------ | ----------------- | -------------------- | ------------ |
| GPT-4o            | 85-95%       | 20-40s             | $0.05-0.15        | Excellent            | Medium       |
| Claude 3.7 Sonnet | 80-90%       | 25-45s             | $0.03-0.10        | Very Good            | Medium       |
| GPT-3.5 Turbo     | 60-75%       | 30-60s             | $0.005-0.02       | Fair                 | Low          |
| Gemini 2.0        | 70-85%       | 25-50s             | $0.02-0.08        | Good                 | Medium       |
| Ollama (Llama3)   | 40-60%       | 15-30s             | $0                | Basic                | High (local) |

> Note: These benchmarks are approximate and vary based on task complexity, implementation details, and hardware configurations.

## Production Optimization Best Practices

Follow these guidelines for optimizing Browser-Use in production:

1. **Browser Configuration**:

   - Use headless mode for all production deployments
   - Set appropriate page load timeouts based on target sites
   - Run in containers with resource limits to prevent overuse

2. **Model Selection**:

   - Use GPT-4o or Claude for critical tasks requiring high accuracy
   - Consider smaller models for simpler tasks or cost efficiency
   - Use local models only for development or privacy-sensitive applications

3. **Resource Management**:

   - Implement connection pooling for LLM API connections
   - Close browser instances properly after use
   - Implement proper error handling and retries

4. **Concurrency Control**:

   - Use semaphores to limit concurrent agents
   - Monitor system resources during execution
   - Set timeouts for all operations

5. **Memory Optimization**:
   - Restrict DOM processing to only what's needed
   - Use viewport expansion judiciously
   - Consider custom browser functions for heavy processing

## Conclusion

Optimizing Browser-Use involves careful consideration of browser configuration, model selection, and resource management. By implementing the techniques described in this chapter, you can significantly improve the performance, reliability, and efficiency of your browser automation tasks.

In the next chapter, we'll explore troubleshooting and debugging techniques for Browser-Use applications.
