import tensorflow as tf


def local_attn(q, k, v):

    def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
        t = x.shape[1]  # x.shape: (16, 4, 64, 256)  (b, windows, window_size, -1)

        paddings = tf.reshape(tf.convert_to_tensor([(0, 0, 0, 1) + (len(x.shape) - 2) * (0, 0)]), shape=(-1, 2))

        padded_x = tf.pad(x, paddings, constant_values=pad_value)
        tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
        return tf.concat(tensors, axis=dim)

    # Collapse num_heads into the batch dimension since computation only happens on the other axes
    merge_into_batch = lambda t: tf.reshape(t, shape=(-1, *t.shape[-2:]))
    q, k, v = map(merge_into_batch, (q, k, v))
    b, t, e = q.shape  # batch * num_heads, time, embedding

    window_size, causal, look_backward, look_forward, shared_qk = 512, True, 1, 0, False
    assert (t % window_size) == 0, f'Sequence length {t} must be divisible by window size {window_size} for local attention'

    windows = t // window_size

    ticker = tf.range(t)[None, :]
    b_t = tf.reshape(ticker, (1, windows, window_size))

    bucket_fn = lambda t: tf.reshape(t, shape=(b, windows, window_size, -1))
    bq, bk, bv = map(bucket_fn, (q, k, v))  # batch * num_heads, num_windows, window_size, embedding dim

    look_around_kwargs = {'backward': look_backward, 'forward': look_forward}
    bk = look_around(bk, **look_around_kwargs)
    bv = look_around(bv, **look_around_kwargs)

    bq_t = b_t
    bq_k = look_around(b_t, **look_around_kwargs)

    dots = tf.einsum('bhie,bhje->bhij', bq, bk) * (e ** -0.5)

    mask_value = -3.4028234663852886e+38

    if causal:
        mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]
        dots = tf.where(mask, dots, mask_value)
        del mask

    mask = bq_k[:, :, None, :] == -1
    dots = tf.where(mask, dots, mask_value)

    w = tf.keras.layers.Activation('softmax', dtype=tf.float32)(dots)
    w = tf.cast(w, v.dtype)

    a = tf.matmul(w, bv)
    a = tf.reshape(a, (8, 2048, 64))

    return a

q = tf.random.normal((8, 2048, 64))
k = tf.random.normal((8, 2048, 64))
v = tf.random.normal((8, 2048, 64))

local_attn(q, k, v)
