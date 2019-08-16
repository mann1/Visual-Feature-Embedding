from . import inject_keras_modules, init_keras_custom_objects
from . import effnet

from .preprocessing import center_crop_and_resize

EfficientNetB0 = inject_keras_modules(effnet.EfficientNetB0)
EfficientNetB1 = inject_keras_modules(effnet.EfficientNetB1)
EfficientNetB2 = inject_keras_modules(effnet.EfficientNetB2)
EfficientNetB3 = inject_keras_modules(effnet.EfficientNetB3)
EfficientNetB4 = inject_keras_modules(effnet.EfficientNetB4)
EfficientNetB5 = inject_keras_modules(effnet.EfficientNetB5)
EfficientNetB6 = inject_keras_modules(effnet.EfficientNetB6)
EfficientNetB7 = inject_keras_modules(effnet.EfficientNetB7)

preprocess_input = inject_keras_modules(effnet.preprocess_input)

init_keras_custom_objects()
