from transformers import ResNetConfig, ResNetModel

def HF_resnet_model(pretrained=True,**kwargs):
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


