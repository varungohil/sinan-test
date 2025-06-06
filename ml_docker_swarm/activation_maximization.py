import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from symbols.cnvnet import get_symbol

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-prefix', dest='model_prefix', type=str, default='./model/cnv')
    parser.add_argument('--load-epoch', dest='load_epoch', type=int, default=200)
    parser.add_argument('--gpus', type=str, default=None, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    parser.add_argument('--output-dir', type=str, default='./activation_maps', help='directory to save activation maps')
    parser.add_argument('--num-iterations', type=int, default=100, help='number of iterations for activation maximization')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate for activation maximization')
    return parser.parse_args()

def load_model(args):
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    
    # Load model parameters
    sym = get_symbol()
    all_layers = sym.get_internals()
    
    # Get the model and internal state
    model = all_layers['latency_output']
    internal_state = all_layers['full_feature_output']
    
    # Create modules with proper data shapes
    data_shapes = [
        ('data1', (1, 6, 28, 5)),  # System data
        ('data2', (1, 5, 5)),      # Latency data
        ('data3', (1, 28))         # Next config data
    ]
    
    # Create modules
    model = mx.mod.Module(
        context=devs,
        symbol=model,
        data_names=['data1', 'data2', 'data3'],
        label_names=None  # No labels needed for activation maximization
    )
    
    internal_state = mx.mod.Module(
        context=devs,
        symbol=internal_state,
        data_names=['data1', 'data2', 'data3'],
        label_names=None
    )
    
    # Initialize the modules
    model.bind(data_shapes=data_shapes, for_training=False)
    internal_state.bind(data_shapes=data_shapes, for_training=False)
    
    # Load parameters
    _, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, args.load_epoch)
    model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
    internal_state.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
    
    return model, internal_state, devs

def clip_system_data(data):
    """Clip system data to valid ranges for each channel"""
    # Create a copy to avoid modifying the original
    clipped = data.copy()
    
    # RPS (channel 0): Non-negative
    clipped[:, 0] = np.maximum(clipped[:, 0], 0)
    
    # Replica count (channel 1): Positive integers
    clipped[:, 1] = np.maximum(np.round(clipped[:, 1]), 1)
    
    # CPU limit (channel 2): Between 0 and 1
    clipped[:, 2] = np.clip(clipped[:, 2], 0, 1)
    
    # CPU usage (channel 3): Between 0 and 1
    clipped[:, 3] = np.clip(clipped[:, 3], 0, 1)
    
    # RSS memory (channel 4): Non-negative
    clipped[:, 4] = np.maximum(clipped[:, 4], 0)
    
    # Cache memory (channel 5): Non-negative
    clipped[:, 5] = np.maximum(clipped[:, 5], 0)
    
    return clipped

def clip_latency_data(data):
    """Clip latency data to valid ranges and apply transformation"""
    # Create a copy to avoid modifying the original
    clipped = data.copy()
    
    # All latency values should be non-negative
    clipped = np.maximum(clipped, 0)
    
    # Apply the same transformation as in training
    d = 505  # Cap value
    k = 0.01  # Smoothing factor
    clipped = np.where(clipped < d, clipped, d + (clipped - d) / (1.0 + k * (clipped - d)))
    
    return clipped

def clip_next_config(data):
    """Clip next configuration data to valid ranges"""
    # Create a copy to avoid modifying the original
    clipped = data.copy()
    
    # CPU limits should be between 0 and 1
    clipped = np.clip(clipped, 0, 1)
    
    return clipped

def maximize_activation(model, layer_name, num_iterations, learning_rate, devs):
    # Initialize random input
    # Handle both CPU and GPU contexts
    if isinstance(devs, list):
        ctx = devs[0]  # GPU case
    else:
        ctx = devs    # CPU case
    
    # Initialize all three inputs with appropriate shapes
    if 'convolution' in layer_name or 'fullyconnected0' in layer_name:
        # System data branch
        data1 = mx.nd.random.uniform(0, 1, shape=(1, 6, 28, 5), ctx=ctx)
        data2 = mx.nd.zeros((1, 5, 5), ctx=ctx)  # Latency data
        data3 = mx.nd.zeros((1, 28), ctx=ctx)    # Next config data
    elif 'fullyconnected1' in layer_name:
        # Latency data branch
        data1 = mx.nd.zeros((1, 6, 28, 5), ctx=ctx)  # System data
        data2 = mx.nd.random.uniform(0, 1, shape=(1, 5, 5), ctx=ctx)
        data3 = mx.nd.zeros((1, 28), ctx=ctx)    # Next config data
    else:
        # Next config or combined branches
        data1 = mx.nd.random.uniform(0, 1, shape=(1, 6, 28, 5), ctx=ctx)  # System data
        data2 = mx.nd.random.uniform(0, 1, shape=(1, 5, 5), ctx=ctx)      # Latency data
        data3 = mx.nd.random.uniform(0, 1, shape=(1, 28), ctx=ctx)        # Next config data
    
    # Attach gradients to the input we're optimizing
    if 'convolution' in layer_name or 'fullyconnected0' in layer_name:
        # System data branch
        data1.attach_grad()
        input_data = data1
    elif 'fullyconnected1' in layer_name:
        # Latency data branch
        data2.attach_grad()
        input_data = data2
    elif 'nxt_fc' in layer_name:
        # Next config branch
        data3.attach_grad()
        input_data = data3
    else:
        # Combined layers - optimize all inputs
        data1.attach_grad()
        data2.attach_grad()
        data3.attach_grad()
        # We'll optimize data3 as the primary input, but all will be updated
        input_data = data3
    
    # Get the symbol from the model
    sym = model.symbol
    all_layers = sym.get_internals()
    
    # Print all available layer names for debugging
    print("\nAvailable layer names:")
    for name in all_layers.list_outputs():
        print(f"  - {name}")
    print()
    
    # For ReLU layers, we want to maximize the pre-activation (before ReLU)
    if 'act' in layer_name:
        # Get the layer name before activation (e.g., 'sys_bn1' from 'sys_act1')
        pre_act_name = layer_name.replace('act', 'bn')
        try:
            layer_output = all_layers[pre_act_name]
        except ValueError:
            print(f"Warning: Could not find layer {pre_act_name}, trying {layer_name}")
            layer_output = all_layers[layer_name]
    else:
        try:
            layer_output = all_layers[layer_name]
        except ValueError:
            # Try alternative naming patterns
            alt_names = [
                f"{layer_name}_output",
                f"{layer_name}_fwd",
                f"{layer_name}_0",
                layer_name.replace('conv', 'convolution'),
                layer_name.replace('fc', 'fullyconnected')
            ]
            for alt_name in alt_names:
                try:
                    layer_output = all_layers[alt_name]
                    print(f"Found layer with alternative name: {alt_name}")
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Could not find layer {layer_name} or any alternatives")
    
    # Determine which inputs are needed for this layer
    layer_args = layer_output.list_arguments()
    data_names = []
    data_shapes = []
    
    if 'data1' in layer_args:
        data_names.append('data1')
        data_shapes.append(('data1', (1, 6, 28, 5)))
    if 'data2' in layer_args:
        data_names.append('data2')
        data_shapes.append(('data2', (1, 5, 5)))
    if 'data3' in layer_args:
        data_names.append('data3')
        data_shapes.append(('data3', (1, 28)))
    
    print(f"Layer {layer_name} requires inputs: {data_names}")
    
    # Create a new module for the target layer using just the target layer's output
    target_module = mx.mod.Module(
        symbol=layer_output,  # Use just the target layer's output
        context=ctx,
        data_names=data_names,  # Only include the inputs that the layer actually needs
        label_names=None
    )
    
    # Bind the module with the input shapes
    target_module.bind(
        data_shapes=data_shapes,
        for_training=True
    )
    
    # Copy parameters from the original model
    target_module.set_params(*model.get_params())
    
    # Create an optimizer
    optimizer = mx.optimizer.SGD(learning_rate=learning_rate)
    updater = mx.optimizer.get_updater(optimizer)
    
    # Initialize lists to store iteration and loss values
    iterations = []
    losses = []
    
    # Maximize activation
    for i in range(num_iterations):
        with mx.autograd.record():
            # Forward pass through the target layer
            # Only pass the inputs that the layer needs
            inputs = []
            if 'data1' in data_names:
                inputs.append(data1)
            if 'data2' in data_names:
                inputs.append(data2)
            if 'data3' in data_names:
                inputs.append(data3)
            
            target_module.forward(mx.io.DataBatch(inputs))
            
            # Get the output of the target layer
            target_output = target_module.get_outputs()[0]  # Now it's the first (and only) output
            
            # Compute loss to maximize the mean activation of the target layer
            loss_val = -mx.nd.mean(target_output)
        
        # Backward pass
        loss_val.backward()
        
        # Update the input data
        updater(0, input_data.grad, input_data)
        
        # Apply constraints based on input type
        if 'convolution' in layer_name or 'fullyconnected0' in layer_name:
            # System data constraints
            input_data = mx.nd.array(clip_system_data(input_data.asnumpy()), ctx=ctx)
        elif 'fullyconnected1' in layer_name:
            # Latency data constraints
            input_data = mx.nd.array(clip_latency_data(input_data.asnumpy()), ctx=ctx)
        else:
            # Next config constraints
            input_data = mx.nd.array(clip_next_config(input_data.asnumpy()), ctx=ctx)
        
        # Store iteration and loss
        iterations.append(i)
        losses.append(loss_val.asscalar())
        
        if i % 10 == 0:
            print(f'Iteration {i}, Loss: {loss_val.asscalar()}')
    
    return input_data, iterations, losses

def visualize_activation(input_data, layer_name, output_dir, iterations=None, losses=None):
    # Convert to numpy array
    data = input_data.asnumpy()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize system data (data1)
    if len(data.shape) == 4:  # (batch, channels, height, width)
        channel_names = ['RPS', 'Replica Count', 'CPU Limit', 'CPU Usage', 'RSS Memory', 'Cache Memory']
        for i in range(data.shape[1]):
            plt.figure(figsize=(10, 5))
            plt.imshow(data[0, i], cmap='viridis')
            plt.colorbar()
            plt.title(f'{layer_name} - {channel_names[i]}')
            plt.savefig(f'{output_dir}/{layer_name}_channel_{i}.png')
            plt.close()
    
    # Visualize latency data (data2)
    elif len(data.shape) == 3:  # (batch, features, time)
        percentile_names = ['90th', '95th', '98th', '99th', '99.9th']
        plt.figure(figsize=(10, 5))
        for i in range(data.shape[1]):
            plt.plot(data[0, i], label=f'{percentile_names[i]} percentile')
        plt.title(f'{layer_name} - Latency Features')
        plt.xlabel('Time Step')
        plt.ylabel('Latency (ms)')
        plt.legend()
        plt.savefig(f'{output_dir}/{layer_name}_latency.png')
        plt.close()
    
    # Visualize next config data (data3)
    elif len(data.shape) == 2:  # (batch, features)
        plt.figure(figsize=(15, 5))
        plt.bar(range(data.shape[1]), data[0])
        plt.title(f'{layer_name} - CPU Limits for Services')
        plt.xlabel('Service Index')
        plt.ylabel('CPU Limit')
        plt.ylim(0, 1)
        plt.savefig(f'{output_dir}/{layer_name}_next_config.png')
        plt.close()
    
    # Visualize loss curve if iterations and losses are provided
    if iterations is not None and losses is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(iterations, losses)
        plt.title(f'{layer_name} - Loss Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'{output_dir}/{layer_name}_loss_curve.png')
        plt.close()

def main():
    args = parse_args()
    
    # Load model
    print("\nLoading model...")
    model, internal_state, devs = load_model(args)
    print("Model loaded successfully!")
    
    # Define layers to analyze - using correct layer names from symbol file
    layers_to_analyze = [
        # System data branch trainable layers
        ('convolution0_output', (1, 6, 28, 5)),    # First conv layer
        ('convolution1_output', (1, 16, 28, 5)),   # Second conv layer
        ('convolution2_output', (1, 16, 28, 5)),   # Third conv layer
        ('convolution3_output', (1, 16, 28, 5)),   # Fourth conv layer
        ('convolution4_output', (1, 32, 28, 5)),   # Fifth conv layer
        ('convolution5_output', (1, 32, 28, 5)),   # Sixth conv layer
        ('fullyconnected0_output', (1, 32)),       # System FC layer
        
        # Latency data branch trainable layers
        ('fullyconnected1_output', (1, 5, 5)),     # Latency FC layer
        
        # Next config branch trainable layers
        ('nxt_fc_output', (1, 28)),                # First next config FC layer
        ('nxt_fc_1_output', (1, 32)),              # Second next config FC layer
        
        # Combined layers trainable layers
        ('fc1_output', (1, 64)),                   # First combined FC layer
        ('fc2_output', (1, 32)),                   # Second combined FC layer
        ('fc3_output', (1, 64)),                   # Third combined FC layer
        ('fc4_output', (1, 64))                    # Output FC layer
    ]
    
    total_layers = len(layers_to_analyze)
    print(f"\nStarting activation maximization for {total_layers} trainable layers...")
    
    # Perform activation maximization for each layer
    for layer_idx, (layer_name, input_shape) in enumerate(layers_to_analyze, 1):
        print(f"\n{'='*80}")
        print(f"Processing layer {layer_idx}/{total_layers}: {layer_name}")
        print(f"Layer type: {'Convolutional' if 'convolution' in layer_name else 'Fully Connected'}")
        print(f"Input shape: {input_shape}")
        print(f"{'='*80}")
        
        input_data, iterations, losses = maximize_activation(
            model, 
            layer_name, 
            args.num_iterations, 
            args.learning_rate,
            devs
        )
        
        # Visualize the results
        print(f"\nGenerating visualizations for {layer_name}...")
        visualize_activation(input_data, layer_name, args.output_dir, iterations, losses)
        print(f"Visualizations saved to {args.output_dir}/{layer_name}_*.png")
        
        # Print final loss
        final_loss = losses[-1]
        print(f"Final loss for {layer_name}: {final_loss:.4f}")
    
    print(f"\n{'='*80}")
    print("Activation maximization completed for all layers!")
    print(f"Results saved in: {args.output_dir}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main() 