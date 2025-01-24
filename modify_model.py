import torch
import torch.nn as nn

import torch
import torch.nn as nn

def forward_hook_factory(name):
    """
    Creates a forward hook function that prints the layer name along with input and output shapes.
    
    Args:
        name (str): The fully qualified name of the module.
    
    Returns:
        function: The hook function.
    """
    def forward_hook(module, input, output):
        # Handle multiple inputs
        if isinstance(input, (tuple, list)):
            input_shapes = [tuple(i.shape) for i in input if isinstance(i, torch.Tensor)]
        elif isinstance(input, torch.Tensor):
            input_shapes = [tuple(input.shape)]
        else:
            input_shapes = ["Multiple Inputs"]

        # Handle multiple outputs
        if isinstance(output, torch.Tensor):
            output_shapes = [tuple(output.shape)]
        elif isinstance(output, (tuple, list)):
            output_shapes = [tuple(o.shape) for o in output if isinstance(o, torch.Tensor)]
        else:
            output_shapes = ["Multiple Outputs"]

        # Print the information
        print(f"{name:<50} | Input Shape: {input_shapes} | Output Shape: {output_shapes}")
    
    return forward_hook

def register_hooks(model):
    """
    Registers forward hooks on all leaf modules of the model.
    
    Args:
        model (nn.Module): The PyTorch model to inspect.
    
    Returns:
        list: A list of hook handles for later removal.
    """
    hooks = []
    for name, module in model.named_modules():
        # Identify leaf modules (modules without children)
        if len(list(module.children())) == 0:
            hook = module.register_forward_hook(forward_hook_factory(name))
            hooks.append(hook)
    return hooks



from collections import OrderedDict
import torch.nn as nn

# note we can also use a recurisve apporach by iterating thru named_modules()
def get_model_upto_layer(model, target_layer_name):
    """
    Returns a new model containing all layers from the given model up to and including the specified target layer.

    Args:
        model (nn.Module): The original model.
        target_layer_name (str): Fully qualified name of the target layer.

    Returns:
        nn.Sequential: A new model containing layers up to and including the target layer.
    """
    layers = OrderedDict()
    target_reached = False

    # Define a helper function to modify containers
    def modify_container(container, full_name_parts):
        """
        Modifies a container to include layers up to and including the target layer.
        
        Args:
            container (nn.Module): The container module to modify.
            full_name_parts (list): Remaining parts of the full layer name within the container.
        
        Returns:
            nn.Sequential: A new container with layers up to and including the target layer.
        """
        sub_layers = OrderedDict()
        target_layer_name = full_name_parts[0]
        remaining_name_parts = full_name_parts[1:]
        for name, layer in container.named_children():
            if name != target_layer_name:
                sub_layers[name] = layer
     
            else:
                if remaining_name_parts:
                    sub_layers[name] = modify_container(layer, remaining_name_parts)
                else:
                    sub_layers[name] = layer
                break

        return nn.Sequential(sub_layers)

    for full_name, layer in model.named_children():

        if target_reached:
            break

        if full_name == target_layer_name:
            target_reached = True

   
        if target_layer_name.startswith(full_name) and full_name != target_layer_name:

            remaining_name_parts = target_layer_name[len(full_name) + 1:].split('.')
            layers[full_name] = modify_container(layer, remaining_name_parts)
            target_reached = True
            break
        else:

            layers[full_name] = layer

    # Create the new model up to the target layer
    return nn.Sequential(*layers.values())

# Example Usage
if __name__ == "__main__":
    from monai.networks.nets import DenseNet121

    # Example model
    densenet = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        dropout_prob=0.0,
        growth_rate=16
    )

    # Define the target layer
    target_layer = "features.denseblock3.denselayer24.layers.conv1"

    # Get the modified model
    model_upto_layer = get_model_upto_layer(densenet, target_layer)
    count = 0
    for full_name, layer in densenet.named_modules():
        if full_name == 'features.denseblock3.denselayer24.layers.conv1':
            print(count)
        count += 1
    print(count)
   
    

    # Test the modified model
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    output = model_upto_layer(input_tensor)

    print(f"Output shape: {output.shape}")




def inspect_model(model, input_tensor):
    """
    Inspects the model by registering forward hooks on leaf modules,
    performing a forward pass, and printing input/output shapes.
    
    Args:
        model (nn.Module): The PyTorch model to inspect.
        input_tensor (torch.Tensor): The input tensor to pass through the model.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Register hooks
    hooks = register_hooks(model)
    
    # Disable gradient calculations for efficiency
    with torch.no_grad():
        try:
            # Perform a forward pass
            model(input_tensor)
        except Exception as e:
            print(f"Error during forward pass: {e}")
        finally:
            # Remove all hooks
            for hook in hooks:
                hook.remove()