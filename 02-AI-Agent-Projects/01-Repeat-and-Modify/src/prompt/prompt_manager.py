"""
Prompt Manager Class.

- load prompt from 'yaml' setting file.
"""

import os

import yaml
from dataclasses import dataclass
from typing import Dict, List, Any, Union, Optional

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate
)

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
    HumanMessage
)

# ------------------------------------------------------------
# Data Class for Chat Prompt Configuration
# ------------------------------------------------------------
@dataclass
class MessagePrompt:
    """
    Data Class for Chat Prompt
    """
    msg_type: str                                  # message type ('system', 'human', 'ai', 'placeholder')
    content: Optional[str] = None                  # message content
    variable_name: Optional[str] = None            # message variable name for 'MessagePlaceholder'
    optional: bool = False                         # bool value that this message is optional message

# ------------------------------------------------------------
# Data Class for Prompt Configuration
# ------------------------------------------------------------
@dataclass
class PromptConfig:
    """
    Config Data Classs for Prompt
    """
    name: str                                      # prompt name
    description: str                               # prompt description
    prompt_type: str                               # prompt type ('single_msg' or 'multi_msg')
    input_variables: List[str]                     # input variable keys of prompt content
    messages: Optional[List[MessagePrompt]] = None # List of MessagePrompt class
    metadata: Optional[Dict[str, Any]] = None      # metadata of prompt

# ------------------------------------------------------------
# Prompt Factor Class
# ------------------------------------------------------------
class PromptFactory:
    """
    Prompt Factory Class that create prompts according to confinguration in yaml.
    """

    @staticmethod
    def from_config(config: PromptConfig) -> ChatPromptTemplate | BaseMessage:
        """
        Create Prompt Message List according to yaml config.

        Args:
            config (PromptConfig) : Config Data Classs instance
        """

        prompt_type = config.prompt_type

        match prompt_type:
            case 'single_msg':
                msg_config = config.messages[0]
                
                msg_type = msg_config.msg_type
                match msg_type:
                    case 'system':
                        prompt = SystemMessage(content=msg_config.content)
                    case 'ai':
                        prompt = AIMessage(content=msg_config.content)
                    case 'human':
                        prompt = HumanMessage(content=msg_config.content)
            case 'multi_msg':
                messages = []

                for msg_config in config.messages:
                    msg_type = msg_config.msg_type
                    match msg_type:
                        case 'placeholder':
                            messages.append(
                                MessagesPlaceholder(
                                    variable_name=msg_config.variable_name,
                                    optional=msg_config.optional
                                )
                            )
                        case 'ai':
                            messages.append(('ai', msg_config.content))
                        case 'human':
                            messages.append(('human', msg_config.content))
                        case 'system':
                            messages.append(('system', msg_config.content))

                prompt = ChatPromptTemplate.from_messages(messages)
                prompt.input_variables = config.input_variables
            
        return prompt
    

# ------------------------------------------------------------
# Prompt Manager Class
# ------------------------------------------------------------
class PromptManager:
    """
    Main Class for Prompt Management
    """

    def __init__(
        self,
        prompts_dir:str = os.path.join('prompts')
    ) -> None:
        """
        Initialize PromptManager Class.

        Args:
            prompts_dir: Dir path that prompt setting yaml files exists.
        """
        self.prompts_dir = prompts_dir
        if not os.path.isdir(prompts_dir):
            os.mkdir(self.prompts_dir)
        
        self._cache: Dict[str, ChatPromptTemplate] = {}
        
    def load_prompt(
        self,
        prompt_name:str,
        use_cache:bool = True
    ) -> ChatPromptTemplate | PromptTemplate:
        """
        Get ChatPromptTemplate from Yaml config.
        """

        if use_cache and prompt_name in self._cache:
            return self._cache.get(prompt_name, [])
        
        config = self._load_prompt_config(prompt_name)
        prompt = PromptFactory.from_config(config)

        if use_cache:
            self._cache[prompt_name] = prompt
        
        return prompt
        
    
    def _load_prompt_config(
        self,
        prompt_name:str
    ) -> PromptConfig:
        """
        Load Prompt Config from yaml file.
        """
        yaml_path = os.path.join(self.prompts_dir, f"{prompt_name}.yaml")

        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"File name '{yaml_path}' not found. Current Work Dir : {os.getcwd()}")

        with open(yaml_path, 'r', encoding='utf-8') as yml:
            yml_config = yaml.safe_load(yml)

        prompt_type = yml_config.get('prompt_type')

        match prompt_type:
            case 'single_msg':
                messages = []
                msg_data = yml_config.get('messages')[0]
                messages.append(MessagePrompt(**msg_data))

            case 'multi_msg':
                messages = []
                for msg_data in yml_config.get('messages', []):
                    messages.append(MessagePrompt(**msg_data))

        return PromptConfig(
            name=yml_config.get('name'),
            description=yml_config.get('description'),
            input_variables=yml_config.get('input_variables'),
            prompt_type=yml_config.get('prompt_type'),
            messages = messages,
            metadata=yml_config.get('metadata')
        )