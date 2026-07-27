"""SigLIP fixed-payload sender adapter."""

from opensemcom.comparaison.dino_sender import StaticFeatureSender


class SiglipSender(StaticFeatureSender):
    method_name = "siglip"
