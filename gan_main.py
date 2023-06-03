import argparse

import tensorflow as tf

from gan import (
    load_dataset,
    define_cond_generator, define_cond_discriminator,
    CGAN,
    plot_metrics,
)


parser = argparse.ArgumentParser(description='GAN model')
parser.add_argument('--action', type=str, required=True, help='Action to perform')
parser.add_argument('--data-dir', type=str, help='Data directory')
parser.add_argument('--cp-dir', type=str, help='Checkpoint directory')
parser.add_argument('--out-dir', type=str, help='Directory for output files')
parser.add_argument('--cp-path', type=str, help='Path to checkpoint to load')
parser.add_argument('--count', type=int, help='Count of output image for each class')
parser.add_argument('--report-file', type=str, help='Path to report file with metrics of losses')
parser.add_argument('--output-file1', type=str, help='Path of output file')
parser.add_argument('--output-file2', type=str, help='Path of output file')


def main(args):
    action = args.action

    if action == 'train':
        data_dir = args.data_dir
        cp_dir = args.cp_dir
        out_dir = args.out_dir
        report_file = args.report_file

        train(data_dir, cp_dir, out_dir)

    if action == 'generate':
        cp_path = args.cp_path
        out_dir = args.out_dir
        count = args.count

        generate(cp_path, out_dir, count)

    if action == 'plot_metrics':
        report_file = args.report_file
        output_file1 = args.output_file1
        output_file2 = args.output_file2

        plot_metrics(report_file, output_file1, output_file2)


def train(data_dir: str, cp_dir: str, out_dir: str, report_file: str):
    model = CGAN()

    dataset = load_dataset(data_dir)
    model.set_dataset(dataset)

    gen = define_cond_generator()
    gen_opt = tf.keras.optimizers.Adam(1e-4)
    gen_loss = tf.keras.losses.BinaryCrossentropy()

    disc = define_cond_discriminator()
    disc_opt = tf.keras.optimizers.Adam(1e-4)
    disc_loss = tf.keras.losses.BinaryCrossentropy()

    model.set_models(gen, gen_opt, gen_loss, disc, disc_opt, disc_loss)

    model.train(cp_dir, out_dir)

    model.import_losses(report_file)


def generate(cp_path: str, out_dir: str, count: int):
    model = CGAN()

    gen = define_cond_generator()
    gen_opt = tf.keras.optimizers.Adam(1e-4)
    gen_loss = tf.keras.losses.BinaryCrossentropy()

    disc = define_cond_discriminator()
    disc_opt = tf.keras.optimizers.Adam(1e-4)
    disc_loss = tf.keras.losses.BinaryCrossentropy()

    model.set_models(gen, gen_opt, gen_loss, disc, disc_opt, disc_loss)

    model.load_models(cp_path)

    model.generate_all(count, out_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    main(args)
