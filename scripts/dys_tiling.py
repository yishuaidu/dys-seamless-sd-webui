import modules.scripts
import modules.sd_hijack
import modules.shared
import gradio

from modules.processing import process_images
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Optional

# Asymmetric tiling script for stable-diffusion-webui
#
# This script allows seamless tiling to be enabled separately for the X and Y axes.
# When this script is in use, the "Tiling" option in the regular UI is ignored.
class Script(modules.scripts.Script):
    # Override from modules.scripts.Script
    def title(self):
        return "dys_tiling"

    # Override from modules.scripts.Script
    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible
    
    # Override from modules.scripts.Script
    def ui(self, is_img2img):
        with gradio.Accordion("dys_tiling", open=False):
            active = gradio.Checkbox(False, label="Active")
            tileX = gradio.Checkbox(False, label="Tile X")
            tileY = gradio.Checkbox(False, label="Tile Y")
            tileXY= gradio.Checkbox(True, label="Tile XY")
            startStep = gradio.Number(0, label="dispersion", precision=0)
            # stopStep = gradio.Number(-1, label="Stop tiling after step N (-1: Don't stop)", precision=0)
        stopStep = -1
        return [active, tileX, tileY, tileXY, startStep, stopStep]

    # Override from modules.scripts.Script
    def process(self, p, active, tileX, tileY, tileXY, startStep, stopStep):
        if (active):
            # Record tiling options chosen for each axis.
            p.extra_generation_params = {
                "Tile X": tileX,
                "Tile Y": tileY,
                "Tile XY": tileXY,
                "Start Tiling From Step": startStep,
                "Stop Tiling After Step": stopStep,
            }

            # Modify the model's Conv2D layers to perform our chosen tiling.
            self.__hijackConv2DMethods(tileX, tileY, tileXY, startStep, stopStep)
        else:
            # Restore model behaviour to normal.
            self.__restoreConv2DMethods()

    def postprocess(self, *args):
        # Restore model behaviour to normal.
        self.__restoreConv2DMethods()
    

    def __hijackConv2DMethods(self, tileX: bool, tileY: bool, tileXY: bool,  startStep: int, stopStep: int):
        if tileXY:
            tileX=True
            tileY=True

        for layer in modules.sd_hijack.model_hijack.layers:
            if type(layer) == Conv2d:
                layer.padding_modeX = 'circular' if tileX else 'constant'
                layer.padding_modeY = 'circular' if tileY else 'constant'
                layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
                layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
                layer.paddingStartStep = startStep
                layer.paddingStopStep = stopStep
                layer._conv_forward = Script.__replacementConv2DConvForward.__get__(layer, Conv2d)


    def __restoreConv2DMethods(self):
        for layer in modules.sd_hijack.model_hijack.layers:
            if type(layer) == Conv2d:
                layer._conv_forward = Conv2d._conv_forward.__get__(layer, Conv2d)

    def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        step = modules.shared.state.sampling_step
        if ((self.paddingStartStep < 0 or step >= self.paddingStartStep) and (self.paddingStopStep < 0 or step <= self.paddingStopStep)):
            working = F.pad(input, self.paddingX, mode=self.padding_modeX)
            working = F.pad(working, self.paddingY, mode=self.padding_modeY)
        else:
            working = F.pad(input, self.paddingX, mode='constant')
            working = F.pad(working, self.paddingY, mode='constant')
        return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)