import logging
import discord
from discord.ext import commands, tasks
import asyncio
import aiohttp
import random
import re
from collections import deque
from typing import Callable, Dict, Tuple, Any, List, get_type_hints
import json

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenRouter API Key and other necessary constants

OPENROUTER_API_KEY=""
DISCORD_TOKEN=""
# Maximum number of messages to keep in conversation history
MAX_CONVERSATION_LENGTH = 30

# Initialize the bot
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# AI thread storage
AI_THREADS = {}

# Tool registry to map tool names to their respective metadata and functions
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Decorator to register a function as a tool
def tool(name: str, description: str):
    def decorator(func: Callable):
        param_types = get_type_hints(func)
        param_names = list(param_types.keys())
        
        TOOL_REGISTRY[name] = {
            "function": func,
            "description": description,
            "param_names": param_names,
            "param_types": param_types,
        }

        logger.info(f"Registered tool: {name}")
        return func
    return decorator

# Define tool functions using the @tool decorator
@tool(name="sum", description="Calculates the sum of two numbers. Usage: sum(a, b) where a and b are numbers.")
async def sum_numbers(a: float, b: float) -> float:
    return a + b

@tool(name="squareRoot", description="Calculates the square root of a number. Usage: squareRoot(x) where x is a non-negative number.")
async def square_root(x: float) -> float:
    if x < 0:
        raise ValueError("Cannot calculate square root of a negative number")
    return x ** 0.5

# Function to generate system prompt instructions for the AI
def generate_system_prompt() -> str:
    instructions = """You are an AI assistant in a Discord server. Respond to the best of your abilities. You have access to various tools that you can use to perform calculations or other tasks. Here are a few examples of how to use these tools:

Example 1:
Human (John): What's 15 plus 27?
AI: To calculate the sum of 15 and 27, I'll use the sum tool.
sum(15, 27)
The result of sum(15, 27) is 42.
So, John, 15 plus 27 equals 42.

Example 2:
Human (Alice): Can you tell me the square root of 64?
AI: Certainly, Alice! I'll use the squareRoot tool to calculate this for you.
squareRoot(64)
The result of squareRoot(64) is 8.
Alice, the square root of 64 is 8.

Example 3:
Human (Bob): If I have 3 apples and my friend gives me 5 more, how many do I have in total?
AI: Let's use the sum tool to calculate this, Bob.
sum(3, 5)
The result of sum(3, 5) is 8.
Bob, you would have 8 apples in total.

Now, here are the tools available for you to use:

"""
    for name, meta in TOOL_REGISTRY.items():
        instructions += f"- {meta['description']}\n"
    instructions += "\nWhen you need to use a tool, use the exact syntax provided in the tool description. Always show your work by using the tools explicitly in your response. Address users by their names when responding to them."
    return instructions

# Print system prompt to console
print("System Prompt:")
print(generate_system_prompt())

# Asynchronous function to interact with OpenRouter API
async def get_openrouter_response(conversation: List[Dict[str, str]]):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_APP_NAME,
        "Content-Type": "application/json",
    }
    system_prompt = generate_system_prompt()
    data = {
        "model": "nousresearch/hermes-3-llama-3.1-405b",
        "messages": [{"role": "system", "content": system_prompt}] + conversation,
        "min_p": 0.1,
        "temperature": 1.1
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                response_json = await response.json()
                ai_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                logger.info(f"AI response received")
                return ai_response
            else:
                error_message = await response.text()
                logger.error(f"Error from OpenRouter: {response.status} - {error_message}")
                return None

# Function to extract tool and parameters from the text based on registered tools
def extract_tool_and_params(text: str) -> Tuple[str, Tuple]:
    tool_pattern = r"(\w+)\(([^)]*)\)"
    match = re.search(tool_pattern, text)

    if match:
        tool_name = match.group(1)
        if tool_name in TOOL_REGISTRY:
            logger.info(f"Tool detected in response: {tool_name}")
            param_str = match.group(2)
            params = param_str.split(",")
            
            tool_info = TOOL_REGISTRY[tool_name]
            param_types = list(tool_info["param_types"].values())
            
            converted_params = []
            for param, param_type in zip(params, param_types):
                try:
                    converted_params.append(param_type(param.strip()))
                except ValueError:
                    logger.error(f"Invalid parameter type for {param}. Expected {param_type}.")
                    raise Exception(f"Invalid parameter type for {param}. Expected {param_type}.")
            
            return tool_name, tuple(converted_params)
    return "", ()

# Function to add a message to the conversation history
def add_message_to_history(channel_id, message, author_name):
    if channel_id not in AI_THREADS:
        AI_THREADS[channel_id] = deque(maxlen=MAX_CONVERSATION_LENGTH)
    AI_THREADS[channel_id].append({"role": message["role"], "content": f"{author_name}: {message['content']}"})

# Async function to manage conversation and tool execution
async def manage_conversation(channel_id):
    conversation = list(AI_THREADS[channel_id])
    
    while True:
        response = await get_openrouter_response(conversation)
        if response is None:
            return "Error communicating with AI service."

        tool_name, params = extract_tool_and_params(response)
        
        if tool_name in TOOL_REGISTRY:
            tool_func = TOOL_REGISTRY[tool_name]["function"]
            logger.info(f"Executing tool: {tool_name}")
            try:
                result = await tool_func(*params)
                add_message_to_history(channel_id, {"role": "assistant", "content": response}, "AI")
                tool_result_message = f"The result of {tool_name}({', '.join(map(str, params))}) is {result}."
                add_message_to_history(channel_id, {"role": "system", "content": tool_result_message}, "System")
                
                # Send the tool result back to the LLM for further processing
                follow_up_response = await get_openrouter_response(conversation)
                if follow_up_response:
                    return follow_up_response
                else:
                    return tool_result_message
            except Exception as e:
                error_message = f"Error executing {tool_name}: {str(e)}"
                logger.error(error_message)
                add_message_to_history(channel_id, {"role": "system", "content": error_message}, "System")
                return error_message
        else:
            return response

# Function to fetch the last 30 messages from a channel
async def fetch_last_30_messages(channel):
    messages = []
    async for message in channel.history(limit=30):
        author_name = message.author.name if message.author != bot.user else "AI"
        messages.append({"role": "user" if message.author != bot.user else "assistant", "content": f"{author_name}: {message.content}"})
    return list(reversed(messages))

# Background task to periodically respond in active channels
@tasks.loop(seconds=30)
async def periodic_response():
    for channel_id in AI_THREADS:
        channel = bot.get_channel(channel_id)
        if channel:
            try:
                response = await manage_conversation(channel_id)
                if response:
                    await channel.send(response)
                    add_message_to_history(channel_id, {"role": "assistant", "content": response}, "AI")
                else:
                    logger.warning(f"No response generated for channel {channel_id}")
            except Exception as e:
                logger.error(f"Error in periodic response for channel {channel_id}: {str(e)}")

# Event listener for when the bot is ready
@bot.event
async def on_ready():
    logger.info(f"We have logged in as {bot.user}")
    periodic_response.start()

# Event listener for when the bot receives a message
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Respond to chat command
    if message.content.lower().startswith("!ai"):
        await handle_ai_command(message)
    
    # Respond to ping
    elif bot.user in message.mentions:
        await handle_ping(message)
    
    # Update conversation history
    channel_id = message.channel.id
    if channel_id in AI_THREADS:
        add_message_to_history(channel_id, {"role": "user", "content": message.content}, message.author.name)

    await bot.process_commands(message)

async def handle_ai_command(message):
    channel_id = message.channel.id
    if channel_id not in AI_THREADS:
        AI_THREADS[channel_id] = deque(maxlen=MAX_CONVERSATION_LENGTH)
        last_messages = await fetch_last_30_messages(message.channel)
        AI_THREADS[channel_id].extend(last_messages)
        await message.channel.send("AI thread started in this channel.")
    
    response = await manage_conversation(channel_id)
    if response:
        await message.channel.send(response)
        add_message_to_history(channel_id, {"role": "assistant", "content": response}, "AI")

async def handle_ping(message):
    channel_id = message.channel.id
    if channel_id not in AI_THREADS:
        AI_THREADS[channel_id] = deque(maxlen=MAX_CONVERSATION_LENGTH)
        last_messages = await fetch_last_30_messages(message.channel)
        AI_THREADS[channel_id].extend(last_messages)
    
    response = await manage_conversation(channel_id)
    if response:
        await message.channel.send(response)
        add_message_to_history(channel_id, {"role": "assistant", "content": response}, "AI")

# Start the bot
bot.run(DISCORD_TOKEN)