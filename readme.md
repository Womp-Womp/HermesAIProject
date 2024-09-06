# Hermes AI Discord Bot

Hermes is an advanced AI-powered Discord bot that allows users to create and interact with multiple AI characters in Discord channels. It uses the OpenRouter API to generate responses and supports various tools and functionalities.

## Features

- Create and manage multiple AI characters
- Start AI threads in specific Discord channels
- Interact with AI characters using natural language
- Support for custom tool packages and functions
- Cross-talk between AI characters (optional)
- View conversation history
- Manage character profiles

## Commands

- `/start_ai`: Start an AI thread for a specific character in a channel
- `/stop_ai`: Stop an AI thread in a specific channel
- `/save_character`: Save a new character persona
- `/delete_character`: Delete a character persona
- `/list_characters`: List all available characters
- `/list_running_ais`: List all running AI threads in a channel
- `/ask_ai`: Directly ask a question to an AI character
- `/view_history`: View the message history for an AI thread
- `/create_tool_package`: Create a new tool package
- `/list_tools`: List all available tools with descriptions
- `/list_tool_packages`: List all available tool packages
- `/toggle_crosstalk`: Toggle crosstalk between characters
- `/aihelp`: Show help information for AI-related commands

## Setup

1. Clone the repository
2. Install the required dependencies (Discord.py, aiohttp, etc.)
3. Set up your OpenRouter API key and Discord bot token
4. Configure the `CHARACTER_PROFILES_FILE` path
5. Run the bot using `python main.py`

## Configuration

Make sure to set the following variables in the script:

- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `YOUR_SITE_URL`: Your website URL
- `YOUR_APP_NAME`: Your application name
- `DISCORD_TOKEN`: Your Discord bot token

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
