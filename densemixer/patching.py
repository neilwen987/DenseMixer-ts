"""
Patching mechanism for the transformers library to enhance MoE implementations.
"""

import logging
import sys
from importlib.util import find_spec

def get_densemixer_logger():
    """Get the densemixer logger with guaranteed console output"""
    logger = logging.getLogger("densemixer")
    
    # Ensure the logger has a console handler
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    
    return logger

logger = get_densemixer_logger()

def apply_patches(config):
    """
    Apply the DenseMixer patch to multiple MoE models in the transformers library.
    
    Args:
        config: Configuration object with enabled status for each model
        
    Returns:
        list: Names of models that were successfully patched
    """
    # Check if transformers is installed
    if find_spec("transformers") is None:
        logger.warning(
            "The transformers library is not installed. The DenseMixer patch will not be applied."
        )
        return []
    
    patched_models = []
    
    # Patch OLMoE if enabled
    if config.is_model_enabled("olmoe"):
        if apply_olmoe_patch(config):
            implementation = "conventional" if config.use_conventional_implementation() else "densemixer"
            patched_models.append(f"OLMoE-{implementation}")
    
    # Patch Qwen2-MoE if enabled
    if config.is_model_enabled("qwen2"):
        if apply_qwen2_moe_patch(config):
            implementation = "conventional" if config.use_conventional_implementation() else "densemixer"
            patched_models.append(f"Qwen2-MoE-{implementation}")

    # Patch Qwen3-MoE if enabled
    if config.is_model_enabled("qwen3"):
        if apply_qwen3_moe_patch(config):
            implementation = "conventional" if config.use_conventional_implementation() else "densemixer"
            patched_models.append(f"Qwen3-MoE-{implementation}")
    
    if patched_models:
        logger.info(f"DenseMixer patches successfully applied to: {', '.join(patched_models)}")
    
    return patched_models


def apply_olmoe_patch(config):
    """Apply patch to OLMoE model"""
    try:
        from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
        
        if config.use_conventional_implementation():
            from .models.olmoe_conventional import ConventionalOlmoeSparseMoeBlock
            # Apply the conventional patch
            OlmoeSparseMoeBlock.forward = ConventionalOlmoeSparseMoeBlock.forward
            logger.info("Successfully patched OLMoE with conventional implementation")
        else:
            from .models.olmoe_custom import CustomOlmoeSparseMoeBlock
            # Apply the DenseMixer patch
            OlmoeSparseMoeBlock.forward = CustomOlmoeSparseMoeBlock.forward
            logger.info("Successfully patched OLMoE with DenseMixer implementation")
        

        return True
    except ImportError as e:
        logger.warning(f"OLMoE module not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error patching OLMoE: {e}")
        return False


def apply_qwen2_moe_patch(config):
    """Apply patch to Qwen2-MoE model"""
    try:
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
        from .models.qwen2_moe_custom import CustomQwen2MoeSparseMoeBlock

        if config.use_conventional_implementation():
            from .models.qwen2_moe_conventional import ConventionalQwen2MoeSparseMoeBlock
            # Apply the conventional patch
            Qwen2MoeSparseMoeBlock.forward = ConventionalQwen2MoeSparseMoeBlock.forward
            logger.info("Successfully patched Qwen2-MoE with conventional implementation")
        else:
            # Apply the DenseMixer patch
            Qwen2MoeSparseMoeBlock.forward = CustomQwen2MoeSparseMoeBlock.forward
            logger.info("Successfully patched Qwen2-MoE with DenseMixer implementation")
        
        return True
    except ImportError as e:
        logger.warning(f"Qwen2-MoE module not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error patching Qwen2-MoE: {e}")
        return False


def apply_qwen3_moe_patch(config):
    """Apply patch to Qwen3-MoE model"""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        from .models.qwen3_moe_custom import CustomQwen3MoeSparseMoeBlock
        
        if config.use_conventional_implementation():
            from .models.qwen3_moe_conventional import ConventionalQwen3MoeSparseMoeBlock
            # Apply the conventional patch
            Qwen3MoeSparseMoeBlock.forward = ConventionalQwen3MoeSparseMoeBlock.forward
            logger.info("Successfully patched Qwen3-MoE with conventional implementation")
        else:
            # Apply the DenseMixer patch
            Qwen3MoeSparseMoeBlock.forward = CustomQwen3MoeSparseMoeBlock.forward
            logger.info("Successfully patched Qwen3-MoE with DenseMixer implementation")

        return True
    except ImportError as e:
        logger.warning(f"Qwen3-MoE module not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error patching Qwen3-MoE: {e}")
        return False