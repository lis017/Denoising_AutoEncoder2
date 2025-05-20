import tensorflow as tf


#업로드 테스트용 주석
class AutoEncoder:
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.en_decoder = None
        self.relu = tf.keras.activations.relu
        self.tanh = tf.keras.activations.tanh
        self.input_output_dim = 784
        self.encoder_hidden_layers = [200, 200]
        self.decoder_hidden_layers = [200, 200]
        self.code_dim = 32

    def build_model(self):

        # Build Encoder
        encoder_input = tf.keras.layers.Input(shape=(self.input_output_dim,), dtype=tf.float32)
        encoder_h_layer = encoder_input
        for dim in self.encoder_hidden_layers:
            encoder_h_layer = tf.keras.layers.Dense(
                units=dim, activation=self.relu, use_bias=True
            )(encoder_h_layer)
        code = tf.keras.layers.Dense(
            units=self.code_dim, activation=self.tanh, use_bias=True
        )(encoder_h_layer)
        self.encoder = tf.keras.models.Model(inputs=encoder_input, outputs=code)

        # def build_model() 의 내용이어서
        # Build Decoder
        decoder_input = tf.keras.layers.Input(shape=(self.code_dim,), dtype=tf.float32)
        decoder_h_layer = decoder_input
        for dim in self.decoder_hidden_layers:
            decoder_h_layer = tf.keras.layers.Dense(
                units=dim, activation=self.relu, use_bias=True
            )(decoder_h_layer)
        decoder_output = tf.keras.layers.Dense(
            units=self.input_output_dim, activation=None, use_bias=True
        )(decoder_h_layer)
        self.decoder = tf.keras.models.Model(inputs=decoder_input, outputs=decoder_output)

        # En-Decoder
        vae_output = self.decoder(code)
        self.en_decoder = tf.keras.models.Model(inputs=encoder_input, outputs=vae_output)
        optimizer_alg = tf.keras.optimizers.Adam(learning_rate=0.001)
        mse = tf.keras.losses.mse
        self.en_decoder.compile(optimizer=optimizer_alg, loss=mse)

    def fit(self, x, y, batch_size, epochs):
        self.en_decoder.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

    def save_weights(self, save_path):
        self.en_decoder.save_weights(save_path)

    def load_weights(self, load_path):
        self.en_decoder.load_weights(load_path)