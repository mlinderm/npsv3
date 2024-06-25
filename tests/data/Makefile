%.sv.vcf.gz : %.vcf.gz
	bcftools view \
		-i '(SVTYPE ~ "DEL" && INFO/SVLEN <= -50 && INFO/SVLEN >= -15000000) || (SVTYPE ~ "INS" && INFO/SVLEN >= 50 && INFO/SVLEN <= 15000000)' \
        -Oz -o $@ $<
	bcftools index -t $@