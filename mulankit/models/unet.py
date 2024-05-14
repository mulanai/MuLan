def UNetSDModel(unet, adapter, replace=False):
    if replace:
        unet.forward = unet.forward_

    unet.forward_ = unet.forward

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        encoder_hidden_states = self.adapter(encoder_hidden_states)
        return unet.forward_(sample, timestep, encoder_hidden_states, **kwargs)

    def func(*args, **kwargs): return forward(unet, *args, **kwargs)
    unet.adapter = adapter
    unet.forward = func
    return unet


def UNetSDXLModel(unet, adapter, replace=False):
    # if replace:
    #     unet.forward = unet.forward_

    # unet.forward_ = unet.forward

    # def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
    #     encoder_hidden_states, text_embeds = self.adapter(encoder_hidden_states, kwargs['added_cond_kwargs']['text_embeds'])
    #     kwargs['added_cond_kwargs']['text_embeds'] = text_embeds
    #     return unet.forward_(sample, timestep, encoder_hidden_states, **kwargs)

    # def func(*args, **kwargs): return forward(unet, *args, **kwargs)
    unet.adapter = adapter
    # unet.forward = func
    return unet
