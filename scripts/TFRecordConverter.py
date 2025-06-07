import os

import hydra
import npsv2_pb2
import tensorflow as tf
import webdataset as wds
from google.protobuf import descriptor_pb2
from omegaconf import OmegaConf
from tqdm import tqdm


def _filename_to_compression(filename: str) -> str | None:
    if filename.endswith(".gz"):
        return "GZIP"
    return None

def _example_image_shape(example):
    return tuple(example.features.feature["image/shape"].int64_list.value)

def _example_sim_images_shape(example):
    if "sim/images/shape" in example.features.feature:
        return tuple(example.features.feature["sim/images/shape"].int64_list.value)
    return (3, 0, None, None, None)

def _extract_metadata_from_first_example(filename, pileup_image_channels=None):
    raw_example = next(
        iter(tf.data.TFRecordDataset(filenames=filename, compression_type=_filename_to_compression(filename)))
    )
    example = tf.train.Example.FromString(raw_example.numpy())

    image_shape = _example_image_shape(example)
    ac, replicates, *sim_image_shape = _example_sim_images_shape(example)
    if replicates > 0:
        assert ac == 3, "Incorrect number of genotypes in simulated data"
        assert image_shape == tuple(sim_image_shape), "Simulated and actual image shapes don't match"
    if pileup_image_channels:
        assert len(pileup_image_channels) <= image_shape[-1], "More channels requested than available"
        image_shape = image_shape[:-1] + (len(pileup_image_channels),)

    return image_shape, replicates

def load_tfrecord_dataset(filename, num_parallel_reads=None, pileup_image_channels=None) -> tf.data.Dataset:
    # Extract image shape from the first example
    shape, _replicates = _extract_metadata_from_first_example(filename, pileup_image_channels=pileup_image_channels)

    proto_features = {
        "variant/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        "image/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        "image/shape": tf.io.FixedLenFeature(shape=(len(shape),), dtype=tf.int64),
        "label": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        "sim/images/shape": tf.io.FixedLenFeature(shape=(len(shape) + 2,), dtype=tf.int64),
        "sim/images/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    }

    def _process_input(proto_string):
        """Helper function for input function that parses a serialized example."""
        parsed_features = tf.io.parse_single_example(serialized=proto_string, features=proto_features)

        features = {
            "variant/encoded": parsed_features["variant/encoded"],
            "image": tf.io.parse_tensor(parsed_features["image/encoded"], tf.uint8),
            "sim/images": tf.io.parse_tensor(parsed_features["sim/images/encoded"], tf.uint8)
        }

        if pileup_image_channels:
            features["image"] = tf.gather(features["image"], indices=list(pileup_image_channels), axis=-1)

        return features, parsed_features["label"]

    compression = _filename_to_compression(filename)

    return tf.data.TFRecordDataset(tf.constant(filename, dtype=tf.string), compression_type=compression).map(_process_input, num_parallel_calls=num_parallel_reads)

def write_webdataset(filename, output_dir, output_filename, reader_threads:int =1) -> None:
    """ Transform NPSV2 TFRecord dataset filname into a WebDataset file output_filename in output_dir"""
    file_descriptor_set = descriptor_pb2.FileDescriptorSet()
    npsv2_pb2.DESCRIPTOR.CopyToProto(file_descriptor_set.file.add())
    descriptor_source = b'bytes://' + file_descriptor_set.SerializeToString()

    output_path = os.path.join(output_dir, f"{output_filename}.tar")
    writer = wds.TarWriter(output_path)

    dataset = load_tfrecord_dataset(filename, num_parallel_reads=reader_threads)
    for features, real_label in tqdm(dataset, desc="Flattening images"):
        _, [contig, start, end, svtype] = tf.io.decode_proto(
            features["variant/encoded"],
            "npsv2.StructuralVariant",
            ["contig", "start", "end", "svtype"],
            [tf.string, tf.int64, tf.int64, tf.int32], # svtype is an enum
            descriptor_source=descriptor_source,
        )

        region = f"{tf.squeeze(contig).numpy().decode('utf-8')}:{tf.squeeze(start)}-{tf.squeeze(end)}"
        key = f"{tf.squeeze(contig).numpy().decode('utf-8')}_{tf.squeeze(start)}_{tf.squeeze(end)}_{npsv2_pb2.StructuralVariant.Type.Name(int(tf.squeeze(svtype)))}"

        sample = {
           "__key__": key,
            "region.txt": region,
            "image.npy.gz": features["image"].numpy(),
            "label.cls": int(real_label),
            "sim.image.npy.gz": features["sim/images"].numpy(),
        }
        writer.write(sample)

    writer.close()


@hydra.main(version_base=None, config_path=".", config_name="TFRecordConverter.yaml")
def main(cfg):
    if OmegaConf.is_missing(cfg, "output"):  # noqa: SIM108
        output = os.getcwd()
    else:
        output = hydra.utils.to_absolute_path(cfg.output)
    write_webdataset(hydra.utils.to_absolute_path(cfg.input), output, "images", reader_threads=cfg.threads)

if __name__ == "__main__":
    main()
