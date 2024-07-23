import os
from collections import deque
import subprocess
from shlex import quote

import pytest

from npsv3.graphs.graph_constructor import GraphConstructor, ReferenceSpan
from npsv3.util.range import Range

from .. import B37_REF_FASTA, HG38_REF_FASTA, data_path

class TestGraphConstructor:
    def test_find_spans(self):
        region = Range("chr1", 0, 100)
        construct = GraphConstructor(
            region, data_path("empty.vcf.gz")
        )
        assert construct.num_spans == 1 and construct.get_span_region(0) == region 

        # SNV in a single span
        start_idx, end_idx = construct.find_spans(Range("chr1", 50, 51))
        assert start_idx == 0 and end_idx == 0

        # INS in a single span
        start_idx, end_idx = construct.find_spans(Range("chr1", 50, 50))
        assert start_idx == 0 and end_idx == 0

        # INS matching existing span
        construct.spans = deque([
            ReferenceSpan(Range("chr1", 0, 50)),
            ReferenceSpan(Range("chr1", 50, 50)), 
            ReferenceSpan(Range("chr1", 50, 100)),
        ])
        start_idx, end_idx = construct.find_spans(Range("chr1", 50, 50))
        assert start_idx == 1 and end_idx == 1, "Should match insertion span"
            
        start_idx, end_idx = construct.find_spans(Range("chr1", 50, 51))
        assert start_idx == 2 and end_idx == 2, "Should match after insertion span"   

        # SNV in a series
        construct.spans = deque([
            ReferenceSpan(Range("chr1", 0, 50)),
            ReferenceSpan(Range("chr1", 50, 51)), 
            ReferenceSpan(Range("chr1", 51, 52)), 
            ReferenceSpan(Range("chr1", 52, 100)),
        ])
        for start, exp_idx in [(49, 0), (50, 1), (51, 2), (52, 3)]:
            start_idx, end_idx = construct.find_spans(Range("chr1", start, start+1))
            assert start_idx == exp_idx and end_idx == exp_idx


    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    def test_graph_construction(self, tmp_path, cfg):
        region = Range("chr1", 8977700, 8977700).expand(cfg.pileup.graph_flank)
        construct = GraphConstructor(
            region, data_path("chr1_8976700_8978700.vcf.gz")
        )
        
        assert sum(len(span.region) for span in construct.spans) == len(region)
        for i, span in enumerate(construct.spans[1:]):
            assert span.region.start == construct.spans[i].region.end

        gfa_path = os.path.join(tmp_path, "test.gfa")
        xg_path = os.path.join(tmp_path, "test.xg") 
        construct.to_gfa(HG38_REF_FASTA, gfa_path)
        
        with open(xg_path, "w") as xg_file:
            convert_command = f"vg convert \
                        --gfa-in {gfa_path} \
                        --xg-out {xg_path}"
            subprocess.run(convert_command, shell=True, stdout=xg_file, check=True)

        gbwt_path = os.path.join(tmp_path, "test.gbwt") 
        gbwt_command = f"vg gbwt \
            --xg-name {quote(xg_path)} \
            --vcf-input {quote(data_path('chr1_8976700_8978700.vcf.gz'))} \
            --vcf-region {quote(str(region))} \
            --ignore-missing \
            --output {quote(gbwt_path)}"
        subprocess.run(gbwt_command, shell=True, check=True)

        threads_command = f"vg paths --extract-gaf --xg {quote(xg_path)} --gbwt {quote(gbwt_path)}"
        with subprocess.Popen(threads_command, shell=True, stdout=subprocess.PIPE, text=True) as threads:
            while True:
                line = threads.stdout.readline()
                if not line and threads.poll() is not None:
                    break
                elif not line:
                    continue
                print(line)


        assert False