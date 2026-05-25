from npsv3._native_graph import Range
from npsv3.images.graph import sample_from_regions

from .. import _create_vcf, data_path

class TestGraphSampling:
 def test_single_del(self, tmp_path, cfg, hg002_sample):
    region = Range("12", 22129564, 22130387)
    sample_from_regions(cfg, { hg002_sample.name: hg002_sample }, data_path("12_22129565_22130387.vcf.gz"))
    