"""Local dataset for Agibot view-transfer training with inpaint control."""

from cosmos_transfer2._src.transfer2.datasets.local_datasets.singleview_dataset import (
    CTRL_TYPE_INFO,
    SingleViewTransferDataset,
)


class AgibotViewTransferDataset(SingleViewTransferDataset):
    """Thin wrapper over SingleViewTransferDataset for inpaint-based view transfer.

    For now this reuses the single-view pipeline and defaults to `control_input_inpaint`.
    A dedicated implementation can later override loading/parsing for custom inpaint data.
    """

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        resolution: str = "720",
        hint_key: str = "control_input_inpaint",
        is_train: bool = True,
        caption_type: str = "t2w_qwen2p5_7b",
        **kwargs,
    ) -> None:
        # SingleViewTransferDataset validates ctrl type from CTRL_TYPE_INFO.
        # Register inpaint here so this dataset can instantiate without changing other datasets.
        CTRL_TYPE_INFO.setdefault("inpaint", {"folder": None})

        super().__init__(
            dataset_dir=dataset_dir,
            num_frames=num_frames,
            video_size=video_size,
            resolution=resolution,
            hint_key=hint_key,
            is_train=is_train,
            caption_type=caption_type,
            **kwargs,
        )
