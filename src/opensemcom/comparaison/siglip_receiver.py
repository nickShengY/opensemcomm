"""SigLIP fixed-transmission receiver adapter."""

from opensemcom.comparaison.dino_receiver import StaticFeatureReceiver


class SiglipReceiver(StaticFeatureReceiver):
    method_name = "siglip"
