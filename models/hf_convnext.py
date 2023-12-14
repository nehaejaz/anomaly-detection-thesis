from transformers import AutoImageProcessor, ConvNextModel, ConvNextConfig

def HF_Convnext(pretrained=True,in_22k=False,**kwargs):
    # Initializing a ConvNext convnext-tiny-224 style configuration
    configuration = ConvNextConfig(
        out_features=["stage1", "stage2", "stage3", "stage4"],
        output_hidden_states=True,
        drop_path_rate=0.7
    )

    model = ConvNextModel.from_pretrained("facebook/convnext-tiny-224", config=configuration)

    # Accessing the model configuration
    configuration = model.config
    # print(model)
    return model

model = HF_Convnext()

# Count the number of trainable parameters - twenty-seven million (27,820,128)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Total number of trainable parameters in convnext-tiny: {total_params}")
