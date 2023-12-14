from transformers import ResNetConfig, ResNetModel

def HF_Resnet(pretrained=True,**kwargs):
    configuration = ResNetConfig(
        hidden_sizes=[64, 128, 256, 512],
        depths=[2, 2, 2, 2],
        layer_type="basic",
        out_features=["stage1", "stage2", "stage3", "stage4"],
        output_hidden_states=True

    )

    model = ResNetModel.from_pretrained("microsoft/resnet-18", config=configuration)
    # Accessing the model configuration
    configuration = model.config
    # print(model)
    # exit()
    return model

model = HF_Resnet()

# Count the number of trainable parameters - eleven million (11,176,512)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Total number of trainable parameters in ResNet-18: {total_params}")


