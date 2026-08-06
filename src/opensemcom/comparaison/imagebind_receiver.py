"""ImageBind fixed-transmission receiver adapter."""

from opensemcom.comparaison.dino_receiver import StaticFeatureReceiver


class ImageBindReceiver(StaticFeatureReceiver):
    method_name = "imagebind"