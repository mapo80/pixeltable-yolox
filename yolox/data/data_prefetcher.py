# Copyright (c) Megvii, Inc. and its affiliates.

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader, device: torch.device):
        self.loader = iter(loader)
        self.device = torch.device(device)
        self.use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.stream = torch.cuda.Stream() if self.use_cuda else None
        self.input_transfer = (
            self._input_cuda_for_image if self.use_cuda else self._input_to_device
        )
        self.record_stream = (
            DataPrefetcher._record_stream_for_image if self.use_cuda else lambda _: None
        )
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                self.input_transfer()
                self.next_target = self.next_target.cuda(non_blocking=True)
        else:
            self.input_transfer()
            self.next_target = self.next_target.to(self.device)

    def next(self):
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None and self.use_cuda:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    def _input_to_device(self):
        self.next_input = self.next_input.to(self.device)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())
