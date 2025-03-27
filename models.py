from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
import os

def get_gpt4o(temperature=0, api_keys=None):
    """Initialize GPT-4o model"""
    key = api_keys.get("openai") if api_keys else None
    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        openai_api_key=key,
    )

def get_gpt4o_mini(temperature=0, api_keys=None):
    """Initialize GPT-4o-mini model"""
    key = api_keys.get("openai") if api_keys else None
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        openai_api_key=key,
    )

def get_gpt4(temperature=0, api_keys=None):
    """Initialize GPT-4 model"""
    key = api_keys.get("openai") if api_keys else None
    return ChatOpenAI(
        model="gpt-4",
        temperature=temperature,
        openai_api_key=key,
    )

def get_gpt4_turbo(temperature=0, api_keys=None):
    """Initialize GPT-4 Turbo model"""
    key = api_keys.get("openai") if api_keys else None
    return ChatOpenAI(
        model="gpt-4-turbo",
        temperature=temperature,
        openai_api_key=key,
    )

def get_gpt4_vision(temperature=0, api_keys=None):
    """Initialize GPT-4 Vision model"""
    key = api_keys.get("openai") if api_keys else None
    return ChatOpenAI(
        model="gpt-4-vision-preview",
        temperature=temperature,
        openai_api_key=key,
    )

def get_gpt35_turbo(temperature=0, api_keys=None):
    """Initialize GPT-3.5 Turbo model"""
    key = api_keys.get("openai") if api_keys else None
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
        openai_api_key=key,
    )

def get_gpt35_turbo_16k(temperature=0, api_keys=None):
    """Initialize GPT-3.5 Turbo 16k context model"""
    key = api_keys.get("openai") if api_keys else None
    return ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=temperature,
        openai_api_key=key,
    )

def get_claude_opus(temperature=0, api_keys=None):
    """Initialize Claude 3 Opus model"""
    key = api_keys.get("anthropic") if api_keys else None
    return ChatAnthropic(
        model="claude-3-opus-20240229",
        temperature=temperature,
        anthropic_api_key=key,
    )

def get_claude_sonnet(temperature=0, api_keys=None):
    """Initialize Claude 3 Sonnet model"""
    key = api_keys.get("anthropic") if api_keys else None
    return ChatAnthropic(
        model="claude-3-sonnet-20240229",
        temperature=temperature,
        anthropic_api_key=key,
    )

def get_claude_haiku(temperature=0, api_keys=None):
    """Initialize Claude 3 Haiku model"""
    key = api_keys.get("anthropic") if api_keys else None
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=temperature,
        anthropic_api_key=key,
    )

def get_gemini_1_5_pro(temperature=0, api_keys=None):
    """Initialize Gemini 1.5 Pro model"""
    key = api_keys.get("google") if api_keys else None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=temperature,
        google_api_key=key,
    )

def get_gemini_1_5_flash(temperature=0, api_keys=None):
    """Initialize Gemini 1.5 flash model"""
    key = api_keys.get("google") if api_keys else None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=temperature,
        google_api_key=key,
    )

def get_gemini_2_0_pro(temperature=0, api_keys=None):
    """Initialize Gemini 2.0 pro model"""
    key = api_keys.get("google") if api_keys else None
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-pro",
        temperature=temperature,
        google_api_key=key,
    )

def get_gemini_2_0_flash(temperature=0, api_keys=None):
    """Initialize Gemini 2.0 flash model"""
    key = api_keys.get("google") if api_keys else None
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
        google_api_key=key,
    )

def get_mistral_large(temperature=0, api_keys=None):
    """Initialize Mistral Large model"""
    key = api_keys.get("mistral") if api_keys else None
    return ChatMistralAI(
        model="mistral-large-latest",
        temperature=temperature,
        mistral_api_key=key,
    )

def get_mistral_medium(temperature=0, api_keys=None):
    """Initialize Mistral Medium model"""
    key = api_keys.get("mistral") if api_keys else None
    return ChatMistralAI(
        model="mistral-medium-latest",
        temperature=temperature,
        mistral_api_key=key,
    )

def get_mistral_small(temperature=0, api_keys=None):
    """Initialize Mistral Small model"""
    key = api_keys.get("mistral") if api_keys else None
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=temperature,
        mistral_api_key=key,
    )

# Map of model providers
MODEL_PROVIDERS = {
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4": "openai",
    "gpt-4-turbo": "openai",
    "gpt-4-vision-preview": "openai",
    "gpt-3.5-turbo": "openai",
    "gpt-3.5-turbo-16k": "openai",
    "claude-opus": "anthropic",
    "claude-sonnet": "anthropic",
    "claude-haiku": "anthropic",
    "gemini-1.5-pro": "google",
    "gemini-1.5-flash": "google",
    "gemini-2.0-pro": "google",
    "gemini-2.0-flash": "google",
    "mistral-large": "mistral",
    "mistral-medium": "mistral",
    "mistral-small": "mistral",
}

# Function to get model by name (convenience function)
def get_model(model_name, temperature=0, api_keys=None):
    """
    Get a model by name
    
    Args:
        model_name: Name of the model to initialize
        temperature: Temperature setting (0-1)
        api_keys: Dictionary of API keys by provider name
            Example: {"openai": "sk-...", "anthropic": "sk-ant-...", 
                     "google": "...", "mistral": "..."}
        
    Returns:
        Initialized model instance
    """
    model_map = {
        # OpenAI models
        "gpt-4o": get_gpt4o,
        "gpt-4o-mini": get_gpt4o_mini,
        "gpt-4": get_gpt4,
        "gpt-4-turbo": get_gpt4_turbo,
        "gpt-4-vision-preview": get_gpt4_vision,
        "gpt-3.5-turbo": get_gpt35_turbo,
        "gpt-3.5-turbo-16k": get_gpt35_turbo_16k,
        
        # Claude models
        "claude-opus": get_claude_opus,
        "claude-sonnet": get_claude_sonnet,
        "claude-haiku": get_claude_haiku,
        
        # Gemini models
        "gemini-1.5-pro": get_gemini_1_5_pro,
        "gemini-1.5-flash": get_gemini_1_5_flash,
        "gemini-2.0-pro": get_gemini_2_0_pro,
        "gemini-2.0-flash": get_gemini_2_0_flash,
        
        # Mistral models
        "mistral-large": get_mistral_large,
        "mistral-medium": get_mistral_medium,
        "mistral-small": get_mistral_small,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Model {model_name} not found. Available models: {list(model_map.keys())}")
    
    return model_map[model_name](temperature=temperature, api_keys=api_keys)