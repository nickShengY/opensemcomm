"""OpenCLIP fixed-transmission receiver adapter."""

from opensemcom.comparaison.dino_receiver import StaticFeatureReceiver


class OpenclipReceiver(StaticFeatureReceiver):
    method_name = "openclip"
