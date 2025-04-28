# 9. Multi-Agent Systems

Browser-Use supports creating multi-agent systems where multiple agents can work together, either in parallel or in coordination. This chapter explores how to implement and manage multi-agent systems for complex browser automation tasks.

## Parallel Agent Execution

Browser-Use can run multiple agents in parallel, each handling different tasks:

```python
# From examples/features/parallel_agents.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig

# Configure a shared browser
browser = Browser(
    config=BrowserConfig(
        disable_security=True,
        headless=False,
        new_context_config=BrowserContextConfig(save_recording_path='./tmp/recordings'),
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
            # Add more tasks as needed
        ]
    ]

    # Run all agents concurrently
    await asyncio.gather(*[agent.run() for agent in agents])

    # Run another agent after the parallel execution is complete
    follow_up_agent = Agent(
        task='Go to apple.com and return the title of the page',
        llm=llm,
        browser=browser,
    )
    await follow_up_agent.run()

    # Clean up
    await browser.close()

asyncio.run(main())
```

Key aspects of parallel agent execution:

- Each agent gets its own browser context automatically
- Agents can run concurrently using `asyncio.gather()`
- A shared browser instance reduces resource usage
- Results can be collected from each agent individually

## Multiple Agents Sharing Browser Context

For use cases where agents need to share browser state, Browser-Use allows multiple agents to work within the same browser context:

```python
# From examples/features/multiple_agents_same_browser.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser

async def main():
    # Create a shared browser
    browser = Browser()

    # Create a shared context
    async with await browser.new_context() as context:
        model = ChatOpenAI(model='gpt-4o')

        # First agent - search for products
        search_agent = Agent(
            task="Go to amazon.com and search for wireless headphones",
            llm=model,
            browser_context=context,  # Share the context
        )
        await search_agent.run()

        # Second agent - continue from where the first left off
        compare_agent = Agent(
            task="Compare the top 3 headphones from the current search results",
            llm=model,
            browser_context=context,  # Same context as first agent
        )
        await compare_agent.run()

        # Third agent - make a purchase decision
        purchase_agent = Agent(
            task="Add the cheapest headphones to the cart",
            llm=model,
            browser_context=context,  # Same context as previous agents
        )
        await purchase_agent.run()

asyncio.run(main())
```

Benefits of shared context:

- Seamless handoff between agents
- Preservation of browser state (cookies, history, etc.)
- Continuation of user journey across agent switches
- More efficient resource usage

## Interactive Multi-Agent System

Browser-Use supports creating interactive systems where agents can be controlled in real-time:

```python
# From examples/features/multiple_agents_same_browser.py
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser

async def main():
    # Create a shared browser and context
    browser = Browser()
    async with await browser.new_context() as context:
        model = ChatOpenAI(model='gpt-4o')
        current_agent = None

        # Function to get user input asynchronously
        async def get_input():
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: input('Enter task (p: pause current agent, r: resume, b: break): ')
            )

        while True:
            # Get user command or new task
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

asyncio.run(main())
```

This example demonstrates an interactive system where users can:

- Enter new tasks to be executed
- Pause the current agent's execution
- Resume a paused agent
- Stop an agent completely

## Specialized Agent Teams

Browser-Use can create teams of specialized agents, each focused on a different aspect of a complex task:

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, Controller, ActionResult

# Create controllers for specialized functions
research_controller = Controller()
analysis_controller = Controller()
reporting_controller = Controller()

# Define specialized actions for each agent type
@research_controller.action("Save research data")
def save_research(data: str):
    with open("research_data.txt", "a") as f:
        f.write(data + "\n")
    return ActionResult(extracted_content="Research data saved")

@analysis_controller.action("Analyze data")
def analyze_data():
    with open("research_data.txt", "r") as f:
        data = f.read()
    # Perform analysis
    analysis = f"Analysis of data: {data[:100]}..."
    with open("analysis_results.txt", "w") as f:
        f.write(analysis)
    return ActionResult(extracted_content=analysis)

@reporting_controller.action("Generate report")
def generate_report():
    with open("analysis_results.txt", "r") as f:
        analysis = f.read()
    # Generate report
    report = f"Report based on analysis: {analysis[:100]}..."
    with open("final_report.txt", "w") as f:
        f.write(report)
    return ActionResult(extracted_content=report)

async def main():
    # Create a shared browser
    browser = Browser()
    llm = ChatOpenAI(model='gpt-4o')

    # Research agent
    research_agent = Agent(
        task="Research the latest advancements in quantum computing on Google Scholar and save the findings",
        llm=llm,
        browser=browser,
        controller=research_controller,
    )
    await research_agent.run()

    # Analysis agent
    analysis_agent = Agent(
        task="Analyze the research data collected on quantum computing to identify key trends",
        llm=llm,
        browser=browser,
        controller=analysis_controller,
    )
    await analysis_agent.run()

    # Reporting agent
    reporting_agent = Agent(
        task="Generate a comprehensive report on quantum computing advancements based on the analysis",
        llm=llm,
        browser=browser,
        controller=reporting_controller,
    )
    await reporting_agent.run()

    await browser.close()

asyncio.run(main())
```

## Collaborative Problem Solving

Browser-Use agents can work together to solve complex problems that require different skills:

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig

async def main():
    # Create browsers for each agent (or share a browser if preferred)
    shared_browser = Browser(config=BrowserConfig(headless=False))
    llm = ChatOpenAI(model='gpt-4o')

    # Define shared storage
    class SharedData:
        problem_statement = ""
        research_findings = ""
        code_solution = ""
        test_results = ""

    data = SharedData()

    # Problem definition agent
    problem_agent = Agent(
        task="Go to https://leetcode.com/problems/two-sum/ and extract the detailed problem statement",
        llm=llm,
        browser=shared_browser,
    )
    result = await problem_agent.run()
    data.problem_statement = result.final_answer

    # Research agent
    research_agent = Agent(
        task=f"Research effective approaches to solve this problem: {data.problem_statement}",
        llm=llm,
        browser=shared_browser,
    )
    result = await research_agent.run()
    data.research_findings = result.final_answer

    # Coding agent
    coding_agent = Agent(
        task=f"""Based on this problem: {data.problem_statement}
                 And these research findings: {data.research_findings}
                 Go to replit.com and write a Python solution to the Two Sum problem""",
        llm=llm,
        browser=shared_browser,
    )
    result = await coding_agent.run()
    data.code_solution = result.final_answer

    # Testing agent
    testing_agent = Agent(
        task=f"""Test this solution to the Two Sum problem:
                 {data.code_solution}
                 Verify it works with multiple test cases.""",
        llm=llm,
        browser=shared_browser,
    )
    result = await testing_agent.run()
    data.test_results = result.final_answer

    await shared_browser.close()

    # Print the final results
    print("Collaborative Solution Process Complete:")
    print(f"Problem: {data.problem_statement[:100]}...")
    print(f"Solution: {data.code_solution[:100]}...")
    print(f"Test Results: {data.test_results[:100]}...")

asyncio.run(main())
```

## Combining Parallel and Sequential Agents

Browser-Use allows you to create complex workflows that combine parallel and sequential agent execution:

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser

async def main():
    browser = Browser()
    llm = ChatOpenAI(model='gpt-4o')

    # First phase: Parallel research across different sources
    research_agents = [
        Agent(
            task=f"Research {topic} on {source} and summarize the findings",
            llm=llm,
            browser=browser,
        )
        for topic, source in [
            ("climate change", "NASA.gov"),
            ("renewable energy", "energy.gov"),
            ("carbon capture", "Wikipedia"),
        ]
    ]

    # Run research agents in parallel
    research_results = await asyncio.gather(*[agent.run() for agent in research_agents])

    # Extract the summaries
    summaries = [result.final_answer for result in research_results]
    combined_research = "\n\n".join(summaries)

    # Second phase: Analysis (sequential)
    analysis_agent = Agent(
        task=f"Analyze these research findings and identify common themes and contradictions:\n{combined_research}",
        llm=llm,
        browser=browser,
    )
    analysis_result = await analysis_agent.run()

    # Third phase: Parallel report creation
    report_agents = [
        Agent(
            task=f"Create a {report_type} report based on this analysis:\n{analysis_result.final_answer}",
            llm=llm,
            browser=browser,
        )
        for report_type in ["technical", "executive summary", "public communication"]
    ]

    # Run report agents in parallel
    report_results = await asyncio.gather(*[agent.run() for agent in report_agents])

    await browser.close()

asyncio.run(main())
```

## Performance Considerations

When working with multi-agent systems, consider these performance tips:

1. **Resource Management**:

   - For many parallel agents, use headless mode to reduce resource usage
   - Monitor system memory when running multiple browser instances
   - Use `keep_alive=True` to maintain browser instances across agent cycles

2. **Coordination Strategies**:

   - Use shared storage (files, databases, or in-memory objects) for data exchange
   - Implement semaphores to limit concurrent agents if resources are constrained
   - Consider using agent hooks to synchronize at critical points

3. **Error Handling**:

   ```python
   async def run_with_error_handling(agent):
       try:
           return await agent.run()
       except Exception as e:
           print(f"Agent error: {str(e)}")
           return None

   # Run with error handling
   results = await asyncio.gather(*[run_with_error_handling(agent) for agent in agents])
   ```

## Conclusion

Multi-agent systems in Browser-Use provide powerful tools for tackling complex web automation tasks. By combining specialized agents, running them in parallel or sequence, and implementing coordination mechanisms, you can create sophisticated automation workflows that accomplish complex goals efficiently.

In the next chapter, we'll explore integration with different LLM models to power Browser-Use agents.
