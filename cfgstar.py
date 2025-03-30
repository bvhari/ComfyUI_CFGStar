import torch


class CFGStar:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model):
        def custom_cfg_function(args):
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x = args["input"]

            dot_product = torch.sum(cond * uncond, dim=(-1, -2))
            squared_norm = torch.sum(uncond * uncond, dim=(-1, -2)) + 1e-8
            s = dot_product / squared_norm
            uncond_scaled = uncond * s[:, :, None, None]

            return x - (uncond_scaled + cond_scale * (cond - uncond_scaled))

        m = model.clone()
        m.set_model_sampler_cfg_function(custom_cfg_function)

        return (m,)


NODE_CLASS_MAPPINGS = {
    "CFGStar": CFGStar,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFGStar": "CFGStar",
}
