"""ImageBind fixed-payload sender adapter."""

from opensemcom.comparaison.dino_sender import StaticFeatureSender


class ImageBindSender(StaticFeatureSender):
    method_name = "imagebind"