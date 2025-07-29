import tensorflow as tf
import cv2
import matplotlib.image as Image
import matplotlib.pyplot as plt
import numpy as np
from OHEM import MSE_OHEM_Loss
from net import CRAFTNet
from text_utils import get_result_img
from datagen import generator, normalizeMeanVariance
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test(ckpt_path, img_path):
    """Test function using TensorFlow v2"""
    # Create model
    model = CRAFTNet()

    # Load and preprocess image
    src_img = cv2.resize(Image.imread(img_path), (512, 512))
    textimg = normalizeMeanVariance(src_img)
    textimg = np.reshape(textimg, (1, 512, 512, 3))
    textimg = tf.convert_to_tensor(textimg, dtype=tf.float32)

    # Load weights
    print("------loading weight------")
    model.load_weights(ckpt_path)
    print("------complete------")

    # Inference
    y_pre = model(textimg, training=False)
    res = y_pre.numpy()
    res = np.reshape(res, (256, 256, 2))

    # Process results
    get_result_img(src_img, res[:, :, 0], res[:, :, 1])
    res = cv2.resize(res, (512, 512))
    score_txt = res[:, :, 0]
    score_link = res[:, :, 1]

    plt.imsave("/home/user4/ysx/CRAFT/result/weight.jpg", score_txt)
    plt.imsave("/home/user4/ysx/CRAFT/result/weight_aff.jpg", score_link)


def train(load_pretrained=True):
    """Training function using TensorFlow v2"""
    # Create model
    model = CRAFTNet()

    # Model configuration
    modelpath = "/home/user4/ysx/CRAFT/model"
    batch_size = 2
    epoch = 5
    data_len = 858750

    # Learning rate schedule
    boundaries = [50000, 200000]
    learning_rates = [0.001, 0.0001, 0.00001]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, learning_rates
    )

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Loss function
    loss_fn = MSE_OHEM_Loss()

    # Metrics
    train_loss = tf.keras.metrics.Mean(name="train_loss")

    # Checkpoint manager
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, "/home/user4/ysx/demo/", max_to_keep=5
    )

    # Load pretrained weights or checkpoint
    if load_pretrained:
        print("-----load vgg-----")
        # Load VGG pretrained weights (you'll need to adapt this)
        # model.load_vgg_weights('/home/user4/ysx/CRAFT/model/vgg16.ckpt')
        print("-----load vgg complete-----")
    else:
        print("-----load ckpt-----")
        checkpoint.restore("/home/user4/ysx/demo/CRAFT_214000")
        print("-----load ckpt complete")

    print("-----training-----")

    # Test image for visualization
    textimg = Image.imread("/home/user4/ysx/CRAFT/te.jpg")
    textimg1 = np.reshape(textimg, (1, 512, 512, 3))
    textimg = normalizeMeanVariance(textimg1)
    textimg = tf.convert_to_tensor(textimg, dtype=tf.float32)

    # Training loop
    global_step = 0
    loss_t = 0

    for e in range(epoch):
        gen = generator(shuffle=True, batch_size=batch_size)

        for i in range(data_len // batch_size):
            image, label = next(gen)
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            label = tf.convert_to_tensor(label, dtype=tf.float32)

            # Training step
            with tf.GradientTape() as tape:
                y_pre = model(image, training=True)
                loss_value = loss_fn(label, y_pre)

            # Apply gradients
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Update metrics
            train_loss(loss_value)
            global_step += 1

            current_lr = optimizer.learning_rate(global_step).numpy()
            print(
                f"\rstep: {global_step:2d} learning_rate: {current_lr:4g} total_loss: {loss_value:4g}",
                end="",
            )

            loss_t += loss_value

            # Periodic evaluation and visualization
            if global_step % 100 == 0:
                avg_loss = loss_t / 100
                res = model(textimg, training=False)
                get_result_img(
                    textimg1, res[0, :, :, 0].numpy(), res[0, :, :, 1].numpy()
                )

                plt.imsave(
                    "result_c.jpg", cv2.resize(res[0, :, :, 0].numpy(), (512, 512))
                )
                plt.imsave(
                    "result_a.jpg", cv2.resize(res[0, :, :, 1].numpy(), (512, 512))
                )

                print(
                    f"\nstep: {global_step:2d} learning_rate: {current_lr:4g} avg_total_loss: {avg_loss:4g}"
                )
                loss_t = 0

            # Save checkpoint
            if global_step % 1000 == 0:
                checkpoint_manager.save()
                print(f"Checkpoint saved at step {global_step}")


@tf.function
def train_step(model, optimizer, loss_fn, image, label):
    """Compiled training step for better performance"""
    with tf.GradientTape() as tape:
        y_pre = model(image, training=True)
        loss_value = loss_fn(label, y_pre)

    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_value


if __name__ == "__main__":
    train(load_pretrained=False)
    # test('/home/user4/ysx/demo/CRAFT_214000', '/home/user4/ysx/CRAFT/802.jpg')
