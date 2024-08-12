import streamlit as st 
import torch 
from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline

device = 'cuda'


def load_model(path):
    weights = torch.load(path)

    model = UNet2DModel(
        sample_size=128, 
        in_channels=3, 
        out_channels=3,
        block_out_channels=(32, 64, 128, 256, 512, 512),
        layers_per_block=2,
        down_block_types=(
            'DownBlock2D', # 128 - 64 - 32 - 16 - 8 - 4 - 2
            'DownBlock2D', 
            'DownBlock2D', 
            'DownBlock2D', 
            'AttnDownBlock2D',
            'AttnDownBlock2D'
        ),
        up_block_types=(
            'AttnUpBlock2D', 
            'AttnUpBlock2D', 
            'UpBlock2D', 
            'UpBlock2D', 
            'UpBlock2D', 
            'UpBlock2D'

        )
    ).to('cuda')
    weights = torch.load(path)
    model.load_state_dict(weights)
    return model 
    

def generate_image(model):
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipe = DDPMPipeline(model, noise_scheduler)
    res = pipe(num_inference_steps=1000)[0]

    # image = transforms.ToTensor()(res[0])
    # plt.imshow(image.permute(1, 2, 0))
    st.image(res[0])

def main():
    path = ''
    st.title('Generate Butterfly images')
    generate = st.button('Generate Butterfly')
    model = load_model(path)
    if generate:
        with st.spinner('Please wait'):
            generate_image(model)


main()
