import os
from dotenv import load_dotenv

def _set_env(key: str):
    """
    Set an environment variable from a .env file.
    If the variable is not found in the .env file, it will not be set.
    
    Parameters:
    key (str): The name of the environment variable to set.
    """
    load_dotenv()
    if key not in os.environ:
        os.environ[key] = os.getenv(key)
    
    value = os.getenv(key)
    # if value:
    #     print(f"{key} has been set to: {value}")
    # else:
    #     print(f"{key} was not found in the .env file.")

    
    

# if __name__ == "__main__":
#     _set_env("OPENAI_API_KEY")