"""OpenCLIP fixed-payload sender adapter."""

from opensemcom.comparaison.dino_sender import StaticFeatureSender


class OpenclipSender(StaticFeatureSender):
    method_name = "openclip"
