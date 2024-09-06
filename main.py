import logging
import discord
from discord.ext import commands
import asyncio
import aiohttp
import random
import json
from collections import deque
from typing import Callable, Dict, Tuple, Any, List, get_type_hints
import re

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Replace with your OpenRouter API Key and other necessary constants
OPENROUTER_API_KEY = ""
YOUR_SITE_URL = "https://yourwebsite.com"
YOUR_APP_NAME = "Hermes"
DISCORD_TOKEN = ""

# File to store character profiles
CHARACTER_PROFILES_FILE = "character_profiles.json"

# Load character profiles from file
def load_character_profiles():
    try:
        with open(CHARACTER_PROFILES_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Character profiles file not found. Creating a new one.")
        return {}

# Save character profiles to file
def save_character_profiles(profiles):
    with open(CHARACTER_PROFILES_FILE, 'w') as f:
        json.dump(profiles, f, indent=2)

# Initialize character profiles
CHARACTER_PROFILES = load_character_profiles()

# Updated structure for AI_THREADS
AI_THREADS = {}
CHANNEL_THREADS = {}

# SETTINGS
MAX_CONVERSATION_LENGTH = 120
ALLOW_CROSSTALK = True

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

        logger.debug(f"Registered tool: {name} with parameters {param_names} and types {param_types}")
        return func
    return decorator

# Define tool functions using the @tool decorator
@tool(name="sum", description="Calculates the sum of two numbers. Usage: sum(a, b) where a and b are numbers.")
async def sum_numbers(a: float, b: float) -> float:
    logger.debug(f"Executing sum_numbers with a={a}, b={b}")
    return a + b

@tool(name="squareRoot", description="Calculates the square root of a number. Usage: squareRoot(x) where x is a non-negative number.")
async def square_root(x: float) -> float:
    logger.debug(f"Executing square_root with x={x}")
    if x < 0:
        raise ValueError("Cannot calculate square root of a negative number")
    return x ** 0.5

# New dictionary to store tool packages
TOOL_PACKAGES = {
    "default": set(TOOL_REGISTRY.keys())
}

# Function to generate system prompt instructions for the AI
def generate_system_prompt(character: str, tool_package: str) -> str:
    prompt = CHARACTER_PROFILES.get(character, "You are an AI assistant. Respond to the best of your abilities.")
    tool_instructions = "\n\nYou have access to the following tools:\n\n"
    for name in TOOL_PACKAGES.get(tool_package, []):
        meta = TOOL_REGISTRY[name]
        tool_instructions += f"- {meta['description']}\n"
    tool_instructions += "\nWhen you need to use a tool, use the exact syntax provided in the tool description. Always show your work by using the tools explicitly in your response."
    return f"You are an AI assistant. Your name is {character}. Do not add your name to messages, it is in the metadata already. Users have given you an extra system prompt; {prompt}. You are in a group chat with other AIs and humans.{tool_instructions}"

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
            
            logger.debug(f"Extracted parameters for {tool_name}: {converted_params}")
            return tool_name, tuple(converted_params)
    return "", ()

# Asynchronous function to interact with OpenRouter API
async def get_openrouter_response(conversation, system_prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_APP_NAME,
        "Content-Type": "application/json",
    }
    data = {
        "model": "nousresearch/hermes-3-llama-3.1-405b",
        "messages": [{"role": "system", "content": system_prompt}] + conversation,
        "min_p": 0.1,
        "temperature": 1.15
    }

    # Log the conversation history and system prompt
    logger.info("Conversation history and system prompt:")
    for msg in data["messages"]:
        logger.info(f"{msg['role']}: {msg['content']}")

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                response_json = await response.json()
                # Log the AI's response
                ai_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                logger.info(f"AI response: {ai_response}")
                return response_json
            else:
                error_message = await response.text()
                logger.error(f"Error from OpenRouter: {response.status} - {error_message}")
                raise Exception(f"Error from OpenRouter: {response.status} - {error_message}")

# Updated function to add a message to the conversation history
def add_message_to_history(channel_id, character, message):
    if channel_id not in AI_THREADS:
        AI_THREADS[channel_id] = {}
    if character not in AI_THREADS[channel_id]:
        system_prompt = generate_system_prompt(character, AI_THREADS[channel_id][character]['tool_package'])
        AI_THREADS[channel_id][character] = {
            'task': None,
            'history': deque(maxlen=MAX_CONVERSATION_LENGTH),
            'system_prompt': system_prompt
        }
    
    if ALLOW_CROSSTALK:
        CHANNEL_THREADS[channel_id]['history'].append(message)
    else:
        AI_THREADS[channel_id][character]['history'].append(message)

# Updated function to manage conversation
async def manage_conversation(channel_id, character):
    thread_data = AI_THREADS[channel_id][character]
    
    conversation = list(thread_data['history'])
    
    if ALLOW_CROSSTALK:
        conversation = CHANNEL_THREADS[channel_id]['history']
        print(f"Conversation: {conversation}")
        if len(conversation) > 0:
            if(conversation[-1]['role'] == f'AI: {character}'):
                logger.info("Bot prevented from responding to itself.")
                return None

    system_prompt = thread_data['system_prompt']
    full_conversation = conversation  # No need to add system prompt here

    # limit conversation to first 5
    if len(full_conversation) > 5:
        full_conversation = full_conversation[-5:]
    
    response = await get_openrouter_response(full_conversation, system_prompt)
    choices = response.get("choices", [])
    
    if not choices:
        logger.warning("No response from OpenRouter.")
        return None

    message_content = choices[0].get("message", {}).get("content", "")
    
    if not message_content.strip():
        logger.warning("Empty response generated.")
        return None

    # Check for tool usage in the response
    tool_name, params = extract_tool_and_params(message_content)
    if tool_name in TOOL_REGISTRY:
        tool_func = TOOL_REGISTRY[tool_name]["function"]
        logger.info(f"Executing tool: {tool_name} with parameters {params}")
        try:
            result = await tool_func(*params)
            logger.debug(f"Tool execution result: {result}")
            tool_result_message = f"The result of {tool_name}({', '.join(map(str, params))}) is {result}."
            add_message_to_history(channel_id, character, {"role": "system", "content": tool_result_message})
            
            # Send the tool result back to the LLM for further processing
            follow_up_response = await get_openrouter_response(full_conversation + [{"role": "system", "content": tool_result_message}], system_prompt)
            if follow_up_response:
                follow_up_content = follow_up_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                return follow_up_content[:2000]  # Limit to 2000 characters for Discord message limit
            else:
                return tool_result_message
        except Exception as e:
            error_message = f"Error executing {tool_name}: {str(e)}"
            logger.error(error_message)
            add_message_to_history(channel_id, character, {"role": "system", "content": error_message})
            return error_message

    return message_content[:2000]  # Limit to 2000 characters for Discord message limit

# Updated background task to monitor messages in a channel for a specific character AI thread
async def monitor_channel(character, channel):
    while True:
        try:
            await asyncio.sleep(random.randint(5,60))  # More reactive: 5-30 seconds
            
            final_message = await manage_conversation(channel.id, character)
        
            if final_message:
                ai_message = f"({character}): {final_message}"
                sent_message = await channel.send(ai_message)
                
                add_message_to_history(channel.id, character, {"role": f'AI: {character}', "content": final_message})
            else:
                logger.warning(f"No message generated for {character} in channel {channel.name}")

        except asyncio.CancelledError:
            logger.info(f"AI thread for {character} in channel {channel.name} has been cancelled.")
            break
        except Exception as e:
            logger.error(f"Error while processing messages in channel {channel.name}: {e}")
            
        if character not in AI_THREADS[channel.id]:
            break 

# Initialize the bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Updated slash command to start an AI thread
@bot.slash_command(name="start_ai", description="Start an AI thread for a specific character in a channel")
async def start_ai(ctx, character: str, channel: discord.Option(discord.TextChannel, description="The channel to start the AI thread in"), tool_package: str = "default"):
    if character in CHARACTER_PROFILES:
        if channel.id not in AI_THREADS:
            AI_THREADS[channel.id] = {}
            CHANNEL_THREADS[channel.id] = {"history": []}
        
        if character in AI_THREADS[channel.id]:
            await ctx.respond(f"An AI thread for {character} is already running in that channel.")
            return
        
        if tool_package not in TOOL_PACKAGES:
            await ctx.respond(f"Tool package '{tool_package}' does not exist. Using default package.")
            tool_package = "default"
        
        system_prompt = generate_system_prompt(character, tool_package)
        task = bot.loop.create_task(monitor_channel(character, channel))
        AI_THREADS[channel.id][character] = {
            'task': task,
            'history': deque(maxlen=MAX_CONVERSATION_LENGTH),
            'system_prompt': system_prompt,
            'tool_package': tool_package
        }
        logger.info(f"Started AI thread for {character} in channel {channel.name} with tool package {tool_package}")
        await ctx.respond(f"AI thread for {character} started in channel {channel.name} with tool package {tool_package}.")
    else:
        await ctx.respond(f"Character '{character}' is not defined. Available characters: {', '.join(CHARACTER_PROFILES.keys())}")

# New slash command to create a tool package
@bot.slash_command(name="create_tool_package", description="Create a new tool package")
async def create_tool_package(ctx, package_name: str, tools: str):
    tool_list = [tool.strip() for tool in tools.split(',')]
    invalid_tools = [tool for tool in tool_list if tool not in TOOL_REGISTRY]
    
    if invalid_tools:
        await ctx.respond(f"Invalid tools: {', '.join(invalid_tools)}. Please use only existing tools.")
        return
    
    TOOL_PACKAGES[package_name] = set(tool_list)
    await ctx.respond(f"Tool package '{package_name}' created with tools: {', '.join(tool_list)}")

# New slash command to list all available tools
@bot.slash_command(name="list_tools", description="List all available tools with descriptions")
async def list_tools(ctx):
    tool_list = "\n".join([f"- {name}: {meta['description']}" for name, meta in TOOL_REGISTRY.items()])
    await ctx.respond(f"Available tools:\n{tool_list}")

# New slash command to list all tool packages
@bot.slash_command(name="list_tool_packages", description="List all available tool packages")
async def list_tool_packages(ctx):
    package_list = "\n".join([f"- {name}: {', '.join(tools)}" for name, tools in TOOL_PACKAGES.items()])
    await ctx.respond(f"Available tool packages:\n{package_list}")

# Updated slash command to stop an AI thread
@bot.slash_command(name="stop_ai", description="Stop an AI thread in a specific channel")
async def stop_ai(ctx, character: str, channel: discord.Option(discord.TextChannel, description="The channel to stop the AI thread in")):
    if channel.id in AI_THREADS and character in AI_THREADS[channel.id]:
        task = AI_THREADS[channel.id][character]['task']
        task.cancel()
        del AI_THREADS[channel.id][character]
        if not AI_THREADS[channel.id]:
            del AI_THREADS[channel.id]
        logger.info(f"Stopped AI thread for {character} in channel {channel.name}")
        await ctx.respond(f"AI thread for {character} in channel {channel.name} has been stopped.")
    else:
        await ctx.respond(f"No AI thread for {character} is running in channel {channel.name}.")

# Slash command to save a new character persona
@bot.slash_command(name="save_character", description="Save a new character persona")
async def save_character(ctx, character_name: str, character_description: str):
    CHARACTER_PROFILES[character_name] = character_description
    save_character_profiles(CHARACTER_PROFILES)
    logger.info(f"Saved new character: {character_name}")
    await ctx.respond(f"Character '{character_name}' has been saved successfully.")

# Slash command to delete a character
@bot.slash_command(name="delete_character", description="Delete a character persona")
async def delete_character(ctx, character_name: str):
    if character_name in CHARACTER_PROFILES:
        del CHARACTER_PROFILES[character_name]
        save_character_profiles(CHARACTER_PROFILES)
        logger.info(f"Deleted character: {character_name}")
        await ctx.respond(f"Character '{character_name}' has been deleted successfully.")
    else:
        await ctx.respond(f"Character '{character_name}' does not exist.")

# Slash command to list all available characters
@bot.slash_command(name="list_characters", description="List all available characters")
async def list_characters(ctx):
    if CHARACTER_PROFILES:
        character_list = "\n".join([f"- {name}: {description[:50]}..." for name, description in CHARACTER_PROFILES.items()])
        await ctx.respond(f"Available characters:\n{character_list}")
    else:
        await ctx.respond("No characters have been defined yet.")

# Slash command to list all running AI threads in a channel
@bot.slash_command(name="list_running_ais", description="List all running AI threads in a channel")
async def list_running_ais(ctx, channel: discord.Option(discord.TextChannel, description="The channel to list running AIs in")):
    if channel.id in AI_THREADS:
        running_ais = ", ".join(AI_THREADS[channel.id].keys())
        await ctx.respond(f"Running AI threads in channel {channel.name}: {running_ais}")
    else:
        await ctx.respond(f"No AI threads are running in channel {channel.name}.")

# Slash command toggle crosstalk
@bot.slash_command(name="toggle_crosstalk", description="Toggle crosstalk between characters")
async def toggle_crosstalk(ctx):
    global ALLOW_CROSSTALK
    ALLOW_CROSSTALK = not ALLOW_CROSSTALK
    await ctx.respond(f"Crosstalk is now {'enabled' if ALLOW_CROSSTALK else 'disabled'}")

# Slash command to directly interact with the AI assistant
@bot.slash_command(name="ask_ai", description="Ask a question to the AI assistant")
async def ask_ai(ctx, character: str, question: str):
    if character not in CHARACTER_PROFILES:
        await ctx.respond(f"Character '{character}' is not defined. Available characters: {', '.join(CHARACTER_PROFILES.keys())}")
        return

    logger.info(f"Received question for {character}: {question}")
    formatted_message = f"({ctx.author.display_name}): {question}"
    add_message_to_history(ctx.channel.id, character, {"role": "user", "content": question})
    response = await manage_conversation(ctx.channel.id, character)
    if response:
        logger.info(f"AI ({character}) response: {response}")
        await ctx.respond(f"({character}): {response}")
    else:
        logger.warning(f"No response generated for {character}")
        await ctx.respond("I'm sorry, but I couldn't generate a response at this time. Please try again later.")

# New slash command to view message history
@bot.slash_command(name="view_history", description="View the message history for an AI thread")
async def view_history(ctx, character: str, channel: discord.Option(discord.TextChannel, description="The channel to view history from")):
    if channel.id in AI_THREADS and character in AI_THREADS[channel.id]:
        history = AI_THREADS[channel.id][character]['history']
        system_prompt = AI_THREADS[channel.id][character]['system_prompt']
        
        history_text = f"System Prompt: {system_prompt}\n\nConversation History:\n"
        for msg in history:
            history_text += f"{msg['role']}: {msg['content']}\n"
        
        # Split the history into chunks of 2000 characters (Discord's message limit)
        chunks = [history_text[i:i+2000] for i in range(0, len(history_text), 2000)]
        
        for chunk in chunks:
            await ctx.respond(chunk)
    else:
        await ctx.respond(f"No AI thread for {character} is running in channel {channel.name}.")

# Updated help command
@bot.slash_command(name="aihelp", description="Show help information for AI-related commands")
async def aihelp(ctx):
    help_embed = discord.Embed(title="AI Bot Help", description="Here are the available commands:", color=0x00ff00)
    
    help_embed.add_field(name="/start_ai", value="Start an AI thread for a specific character in a channel with an optional tool package", inline=False)
    help_embed.add_field(name="/stop_ai", value="Stop an AI thread for a specific character in a channel", inline=False)
    help_embed.add_field(name="/save_character", value="Save a new character persona", inline=False)
    help_embed.add_field(name="/delete_character", value="Delete a character persona", inline=False)
    help_embed.add_field(name="/list_characters", value="List all available characters", inline=False)
    help_embed.add_field(name="/list_running_ais", value="List all running AI threads in a channel", inline=False)
    help_embed.add_field(name="/ask_ai", value="Ask a question to the AI assistant", inline=False)
    help_embed.add_field(name="/view_history", value="View the message history for an AI thread", inline=False)
    help_embed.add_field(name="/create_tool_package", value="Create a new tool package", inline=False)
    help_embed.add_field(name="/list_tools", value="List all available tools with descriptions", inline=False)
    help_embed.add_field(name="/list_tool_packages", value="List all available tool packages", inline=False)
    help_embed.add_field(name="/aihelp", value="Show this help message", inline=False)
    help_embed.add_field(name="/toggle_crosstalk", value="Toggle crosstalk between characters", inline=False)
    
    help_embed.set_footer(text="For more detailed information, please refer to the bot documentation.")
    
    await ctx.respond(embed=help_embed)

# Event listener for when the bot is ready
@bot.event
async def on_ready():
    logger.info(f"We have logged in as {bot.user}")

# Updated event listener for when a message is received
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.channel.id in AI_THREADS:
        for character in AI_THREADS[message.channel.id]:
            add_message_to_history(message.channel.id, character, {"role": f'USER: {message.author.name}', "content": message.content})

    await bot.process_commands(message)

# Start the bot
bot.run(DISCORD_TOKEN)