import numpy as np


# https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
class AnnotatedArray(np.ndarray):
    def __new__(cls, input_array, fisher_strand=None, strand_orientation_bias=None):
        obj = np.asarray(input_array).view(cls)
        # Genomic attributes
        obj.fisher_strand = fisher_strand
        obj.strand_orientation_bias = strand_orientation_bias

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.fisher_strand = getattr(obj, "fisher_strand", None)
        self.strand_orientation_bias = getattr(obj, "strand_orientation_bias", None)
