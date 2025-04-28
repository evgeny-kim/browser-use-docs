# 12. Extension and Integration with Other Services

Browser-Use can be integrated with external services and platforms to extend its functionality beyond standalone applications. This chapter explores how to integrate Browser-Use with messaging platforms like Discord and Slack, enabling browser automation through conversational interfaces.

## Discord Integration

Browser-Use can be integrated with Discord to create bots that perform browser automation tasks in response to messages. The following example demonstrates how to create a Discord bot using Browser-Use:

```python
# From examples/integrations/discord/discord_example.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import BrowserConfig
from examples.integrations.discord.discord_api import DiscordBot

load_dotenv()

# Load credentials from environment variables
bot_token = os.getenv('DISCORD_BOT_TOKEN')
if not bot_token:
    raise ValueError('Discord bot token not found in .env file.')

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError('GEMINI_API_KEY is not set')

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

# Create and run the Discord bot
bot = DiscordBot(
    llm=llm,  # required; instance of BaseChatModel
    prefix='$bu',  # optional; prefix of messages to trigger browser-use, defaults to "$bu"
    ack=True,  # optional; whether to acknowledge task receipt with a message, defaults to False
    browser_config=BrowserConfig(
        headless=False
    ),  # optional; useful for changing headless mode or other browser configs
)

bot.run(
    token=bot_token,  # required; Discord bot token
)
```

### Discord Bot Implementation

The `DiscordBot` class provides a complete implementation for handling Discord messages and executing browser automation tasks:

```python
# From examples/integrations/discord/discord_api.py
import discord
from discord.ext import commands
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from browser_use import BrowserConfig
from browser_use.agent.service import Agent, Browser

class DiscordBot(commands.Bot):
    """Discord bot implementation for Browser-Use tasks.

    This bot allows users to run browser automation tasks through Discord messages.
    Processes tasks asynchronously and sends the result back to the user in response to the message.
    Messages must start with the configured prefix (default: "$bu") followed by the task description.

    Args:
        llm (BaseChatModel): Language model instance to use for task processing
        prefix (str, optional): Command prefix for triggering browser tasks. Defaults to "$bu"
        ack (bool, optional): Whether to acknowledge task receipt with a message. Defaults to False
        browser_config (BrowserConfig, optional): Browser configuration settings.
            Defaults to headless mode
    """

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

        # Initialize the bot
        super().__init__(command_prefix='!', intents=intents)

    async def on_ready(self):
        """Called when the bot is ready."""
        try:
            print(f'We have logged in as {self.user}')
            cmds = await self.tree.sync()  # Sync the command tree with discord
        except Exception as e:
            print(f'Error during bot startup: {e}')

    async def on_message(self, message):
        """Called when a message is received."""
        try:
            if message.author == self.user:  # Ignore the bot's messages
                return
            if message.content.strip().startswith(f'{self.prefix} '):
                if self.ack:
                    try:
                        await message.reply(
                            'Starting browser use task...',
                            mention_author=True,
                        )
                    except Exception as e:
                        print(f'Error sending start message: {e}')

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
        """Run a Browser-Use agent with the given task."""
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

### Setting Up a Discord Bot

To use the Discord integration:

1. Create a Discord Application and Bot:

   - Go to the Discord Developer Portal: https://discord.com/developers/applications
   - Create a new application and configure a bot
   - Enable "Message Content Intent" under "Privileged Gateway Intents"
   - Generate an invite URL with bot permissions
   - Invite the bot to your server

2. Set up environment variables:

   ```
   DISCORD_BOT_TOKEN=your-bot-token
   GEMINI_API_KEY=your-gemini-api-key  # Or any other LLM API key
   ```

3. Run the Discord bot example script
4. Interact with the bot in Discord using the `$bu` prefix:
   ```
   $bu search for the weather in Tokyo
   ```

## Slack Integration

Browser-Use can be integrated with Slack to create bots that perform browser automation tasks via a Slack app. The following example demonstrates how to create a Slack bot using Browser-Use:

```python
# From examples/integrations/slack/slack_example.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import BrowserConfig
from examples.integrations.slack.slack_api import SlackBot, app

load_dotenv()

# Load credentials from environment variables
bot_token = os.getenv('SLACK_BOT_TOKEN')
if not bot_token:
    raise ValueError('Slack bot token not found in .env file.')

signing_secret = os.getenv('SLACK_SIGNING_SECRET')
if not signing_secret:
    raise ValueError('Slack signing secret not found in .env file.')

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError('GEMINI_API_KEY is not set')

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

# Create the Slack bot
slack_bot = SlackBot(
    llm=llm,  # required; instance of BaseChatModel
    bot_token=bot_token,  # required; Slack bot token
    signing_secret=signing_secret,  # required; Slack signing secret
    ack=True,  # optional; whether to acknowledge task receipt with a message, defaults to False
    browser_config=BrowserConfig(
        headless=True
    ),  # optional; useful for changing headless mode or other browser configs
)

app.dependency_overrides[SlackBot] = lambda: slack_bot

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('integrations.slack.slack_api:app', host='0.0.0.0', port=3000)
```

### Slack Bot Implementation

The `SlackBot` class provides a complete implementation for handling Slack events and executing browser automation tasks:

```python
# From examples/integrations/slack/slack_api.py
import logging
from fastapi import Depends, FastAPI, HTTPException, Request
from langchain_core.language_models.chat_models import BaseChatModel
from slack_sdk.errors import SlackApiError
from slack_sdk.signature import SignatureVerifier
from slack_sdk.web.async_client import AsyncWebClient
from browser_use import BrowserConfig
from browser_use.agent.service import Agent, Browser

app = FastAPI()

class SlackBot:
    def __init__(
        self,
        llm: BaseChatModel,
        bot_token: str,
        signing_secret: str,
        ack: bool = False,
        browser_config: BrowserConfig = BrowserConfig(headless=True),
    ):
        if not bot_token or not signing_secret:
            raise ValueError('Bot token and signing secret must be provided')

        self.llm = llm
        self.ack = ack
        self.browser_config = browser_config
        self.client = AsyncWebClient(token=bot_token)
        self.signature_verifier = SignatureVerifier(signing_secret)
        self.processed_events = set()
        logger.info('SlackBot initialized')

    async def handle_event(self, event, event_id):
        """Handle incoming Slack events."""
        try:
            logger.info(f'Received event id: {event_id}')
            if not event_id:
                logger.warning('Event ID missing in event data')
                return

            if event_id in self.processed_events:
                logger.info(f'Event {event_id} already processed')
                return
            self.processed_events.add(event_id)

            if 'subtype' in event and event['subtype'] == 'bot_message':
                return

            text = event.get('text')
            user_id = event.get('user')
            if text and text.startswith('$bu '):
                task = text[len('$bu ') :].strip()
                if self.ack:
                    try:
                        await self.send_message(
                            event['channel'],
                            f'<@{user_id}> Starting browser use task...',
                            thread_ts=event.get('ts')
                        )
                    except Exception as e:
                        logger.error(f'Error sending start message: {e}')

                try:
                    agent_message = await self.run_agent(task)
                    await self.send_message(
                        event['channel'],
                        f'<@{user_id}> {agent_message}',
                        thread_ts=event.get('ts')
                    )
                except Exception as e:
                    await self.send_message(
                        event['channel'],
                        f'Error during task execution: {str(e)}',
                        thread_ts=event.get('ts')
                    )
        except Exception as e:
            logger.error(f'Error in handle_event: {str(e)}')

    async def run_agent(self, task: str) -> str:
        """Run a Browser-Use agent with the given task."""
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
            logger.error(f'Error during task execution: {str(e)}')
            return f'Error during task execution: {str(e)}'

    async def send_message(self, channel, text, thread_ts=None):
        """Send a message to a Slack channel."""
        try:
            await self.client.chat_postMessage(channel=channel, text=text, thread_ts=thread_ts)
        except SlackApiError as e:
            logger.error(f'Error sending message: {e.response["error"]}')

@app.post('/slack/events')
async def slack_events(request: Request, slack_bot: SlackBot = Depends()):
    """Handle incoming Slack events."""
    try:
        # Verify request signature
        if not slack_bot.signature_verifier.is_valid_request(await request.body(), dict(request.headers)):
            logger.warning('Request verification failed')
            raise HTTPException(status_code=400, detail='Request verification failed')

        event_data = await request.json()
        logger.info(f'Received event data: {event_data}')

        # Handle URL verification
        if 'challenge' in event_data:
            return {'challenge': event_data['challenge']}

        # Process events
        if 'event' in event_data:
            try:
                await slack_bot.handle_event(event_data.get('event'), event_data.get('event_id'))
            except Exception as e:
                logger.error(f'Error handling event: {str(e)}')

        return {}
    except Exception as e:
        logger.error(f'Error in slack_events: {str(e)}')
        raise HTTPException(status_code=500, detail='Internal Server Error')
```

### Setting Up a Slack Bot

To use the Slack integration:

1. Create a Slack App:

   - Go to the Slack API: https://api.slack.com/apps
   - Create a new app from scratch
   - Configure Bot Token Scopes:
     - `chat:write`
     - `channels:history`
     - `im:history`
   - Enable Event Subscriptions:
     - Subscribe to bot events: `message.channels`, `message.im`
     - Set up a Request URL (e.g., using ngrok)

2. Install the app to your workspace and invite it to channels

3. Set up environment variables:

   ```
   SLACK_BOT_TOKEN=your-bot-token
   SLACK_SIGNING_SECRET=your-signing-secret
   GEMINI_API_KEY=your-gemini-api-key  # Or any other LLM API key
   ```

4. Run the Slack bot example script
5. Interact with the bot in Slack using the `$bu` prefix:
   ```
   $bu search for the weather in Tokyo
   ```

## Creating Custom Integrations

Browser-Use can be integrated with other services and platforms by following similar patterns to the Discord and Slack integrations. Here are the key components for creating custom integrations:

### Core Integration Components

1. **Event Handling**:

   - Set up listeners for incoming events (messages, API calls, etc.)
   - Extract the task description from the event

2. **Agent Execution**:

   - Initialize a Browser-Use agent with the task
   - Run the agent asynchronously
   - Capture the result

3. **Response Handling**:
   - Format the agent result for the target platform
   - Send the response back to the user

### Integration Template

```python
# Example template for custom integrations
from browser_use import Agent, Browser, BrowserConfig
from langchain_core.language_models.chat_models import BaseChatModel

class CustomIntegration:
    def __init__(
        self,
        llm: BaseChatModel,
        browser_config: BrowserConfig = BrowserConfig(headless=True),
        # Additional parameters specific to the integration
    ):
        self.llm = llm
        self.browser_config = browser_config
        # Initialize platform-specific clients/connections

    async def handle_incoming_event(self, event_data):
        """Handle incoming events from the integration platform."""
        # Extract task from event data
        task = self.extract_task_from_event(event_data)

        # Send acknowledgment if needed
        await self.send_acknowledgment(event_data)

        # Execute the browser task
        try:
            result = await self.run_agent(task)
            await self.send_response(event_data, result)
        except Exception as e:
            await self.send_error(event_data, str(e))

    async def run_agent(self, task: str) -> str:
        """Run a Browser-Use agent with the given task."""
        try:
            browser = Browser(config=self.browser_config)
            agent = Agent(task=task, llm=self.llm, browser=browser)
            result = await agent.run()

            # Extract result content
            final_content = self.extract_result_content(result)

            # Clean up resources
            await browser.close()

            return final_content
        except Exception as e:
            raise Exception(f'Browser-use task failed: {str(e)}')

    def extract_task_from_event(self, event_data):
        """Extract task description from platform-specific event data."""
        # Implementation depends on the platform
        pass

    async def send_acknowledgment(self, event_data):
        """Send acknowledgment to the user."""
        # Implementation depends on the platform
        pass

    async def send_response(self, event_data, result):
        """Send agent result to the user."""
        # Implementation depends on the platform
        pass

    async def send_error(self, event_data, error_message):
        """Send error message to the user."""
        # Implementation depends on the platform
        pass

    def extract_result_content(self, result):
        """Extract content from agent result."""
        if result.is_done():
            return result.history[-1].result[0].extracted_content
        return "Task did not complete successfully."
```

## Best Practices for Integrations

1. **Error Handling**:

   - Implement robust error handling at each stage
   - Provide clear error messages to users
   - Log errors for troubleshooting

2. **Rate Limiting**:

   - Implement rate limiting for API calls
   - Handle concurrent requests appropriately
   - Consider using message queues for high-volume scenarios

3. **Security**:

   - Verify incoming requests using platform-specific mechanisms
   - Securely store and handle API tokens and credentials
   - Use secure connections for all communications

4. **Statelessness**:

   - Design integrations to be stateless where possible
   - Use external storage for persistent data
   - Implement proper cleanup of browser resources

5. **User Experience**:
   - Provide clear feedback about task status
   - Format responses appropriately for the platform
   - Consider threading or conversation context for multi-turn interactions

## Conclusion

Browser-Use can be integrated with various external services and platforms to extend its capabilities. By implementing integrations with messaging platforms like Discord and Slack, you can create conversational interfaces for browser automation, making the technology more accessible to users.

In the next chapter, we'll explore how Browser-Use can be integrated with Jupyter notebooks for interactive development and exploration.
