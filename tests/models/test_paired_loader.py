import pytest
import webdataset as wds

from npsv3.models.paired import split_and_pad_support

from .. import data_path, result_path

@pytest.mark.cfg_overrides(
    f"pileup.image_channels=[0,1,2,3,4,5,6,7]",
)
class TestPairedDataLoader:
    def test_split_and_pad(self, cfg):
        dataset = (
            wds.WebDataset(data_path("images-0000.tar"))
            .decode()
            .to_tuple("image.npy.gz", "sim.images.npy.gz", "label.cls")
            .compose(split_and_pad_support(max_genotypes=6, padding_value=0))
        )

        for _i, sample in enumerate(dataset):
            assert isinstance(sample, tuple)
            query, support, num_support, label = sample
            
            assert query.shape == (cfg.pileup.image_height, cfg.pileup.image_width, len(cfg.pileup.image_channels))
            assert support.shape == (6, cfg.pileup.image_height, cfg.pileup.image_width, len(cfg.pileup.image_channels))
            assert num_support == 4, "Only 4 genotypes in support data"
            assert label == 3
    
        assert _i == 1, "The two replicates become 2 examples in the dataset"