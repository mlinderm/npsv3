REFERENCE_B37 ?= /data/human_g1k_v37.fasta
REFERENCE_HG38 ?= /data/Homo_sapiens_assembly38.fasta

%.sv.vcf.gz : %.vcf.gz
	bcftools view \
		-i '(SVTYPE ~ "DEL" && INFO/SVLEN <= -50 && INFO/SVLEN >= -15000000) || (SVTYPE ~ "INS" && INFO/SVLEN >= 50 && INFO/SVLEN <= 15000000)' \
        -Oz -o $@ $<
	bcftools index -t $@

images-0000.tar:
	npsv3 command=images \
        reference=$(REFERENCE_B37) \
        generator=coverage \
		input=12_22129565_22130387.vcf.gz \
        background=12_22129565_22130387.background.vcf.gz \
		reads=12_22127565_22132387.bam \
		stats_path=stats.json \
		output=$(PWD) \
		simulation.replicates=2 \
		load_reference=true \
		pileup.example_fn=npsv3.images.example.vcf_to_graph_examples \
		pileup.image_channels='[0,1,2,3,4,5,6,7]'
