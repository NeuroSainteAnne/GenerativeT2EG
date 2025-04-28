import tensorflow as tf



bce_loss_object = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2, reduction=tf.keras.losses.Reduction.NONE)
mae_loss_object = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    
def generator_loss(disc_generated_output, pseudo_t2eg, real_t2eg, lambda_l1,
                   pseudo_sobel, real_sobel, lambda_sobel, weights, pooled_weights):
    "GENERATOR loss contains L1 loss, Edge loss, GAN loss"
    gan_loss = tf.reduce_mean(
        bce_loss_object(
            tf.ones_like(disc_generated_output), 
            disc_generated_output, 
            sample_weight=pooled_weights), 
        axis=(1,2))
    l1_loss = mae_loss_object(tf.reshape(real_t2eg*weights, (real_t2eg.shape[0],-1)),
                              tf.reshape(pseudo_t2eg*weights, (pseudo_t2eg.shape[0],-1)))
    edge_loss = mae_loss_object(tf.reshape(real_sobel*weights, (real_sobel.shape[0],-1)),
                              tf.reshape(pseudo_sobel*weights, (pseudo_sobel.shape[0],-1)))
    total_gen_loss = gan_loss + (lambda_l1 * l1_loss) + (lambda_sobel * edge_loss)
    return total_gen_loss, gan_loss, l1_loss, edge_loss

def generator_loss_withpred(disc_generated_output, pseudo_t2eg, real_t2eg, lambda_l1,
                            pseudo_sobel, real_sobel, lambda_sobel,
                            coarse_segm, generated_segm, lambda_segm,
                            coarse_pred, generated_pred, lambda_pred,
                            weights, coarse_weights, pooled_weights):
    "GENERATOR loss contains L1 loss, Edge loss, GAN loss"
    gan_loss = tf.reduce_mean(
        bce_loss_object(
            tf.ones_like(disc_generated_output), 
            disc_generated_output, 
            sample_weight=pooled_weights), 
        axis=(1,2))
    l1_loss = mae_loss_object(tf.reshape(real_t2eg*weights, (real_t2eg.shape[0],-1)),
                              tf.reshape(pseudo_t2eg*weights, (pseudo_t2eg.shape[0],-1)))
    edge_loss = mae_loss_object(tf.reshape(real_sobel*weights, (real_sobel.shape[0],-1)),
                              tf.reshape(pseudo_sobel*weights, (pseudo_sobel.shape[0],-1)))
    segm_loss = bce_loss_object(tf.reshape(coarse_segm*coarse_weights, (coarse_segm.shape[0],-1)),
                              tf.reshape(generated_segm*coarse_weights, (generated_segm.shape[0],-1)))
    pred_loss = bce_loss_object(tf.reshape(coarse_pred*coarse_weights, (coarse_pred.shape[0],-1)),
                              tf.reshape(generated_pred*coarse_weights, (generated_pred.shape[0],-1)))
    total_gen_loss = gan_loss + (lambda_l1 * l1_loss) + (lambda_sobel * edge_loss) + (lambda_segm * segm_loss) + (lambda_pred * pred_loss)
    return total_gen_loss, gan_loss, l1_loss, edge_loss, segm_loss, pred_loss


def discriminator_loss(disc_real_output, disc_generated_output, weights, pooled_weights):
    "DISCRIMINATOR loss contains real and generated loss"
    real_loss = tf.reduce_mean(
        bce_loss_object(
            tf.ones_like(disc_real_output), 
            disc_real_output, 
            sample_weight=pooled_weights), 
        axis=(1,2))
    generated_loss = tf.reduce_mean(
        bce_loss_object(
            tf.zeros_like(disc_generated_output),
            disc_generated_output,
            sample_weight=pooled_weights),
        axis=(1,2))
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss