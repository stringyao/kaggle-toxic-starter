from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model

from steps.keras.models import CharVDCNNTransformer, WordSCNNTransformer, WordCuDNNGRUTransformer, \
    WordCuDNNLSTMTransformer, WordDPCNNTransformer, ClassifierXY
from steps.keras.callbacks import NeptuneMonitor, ReduceLR
from steps.keras.architectures import dropout_block, cudnn_gru_block, classification_block
from steps.utils import create_filepath


def callbacks(**kwargs):
    lr_scheduler = ReduceLR(**kwargs['lr_scheduler'])
    early_stopping = EarlyStopping(**kwargs['early_stopping'])
    checkpoint_filepath = kwargs['model_checkpoint']['filepath']
    create_filepath(checkpoint_filepath)
    model_checkpoint = ModelCheckpoint(**kwargs['model_checkpoint'])
    neptune = NeptuneMonitor(**kwargs['neptune_monitor'])
    return [neptune, lr_scheduler, early_stopping, model_checkpoint]


class CharVDCNN(CharVDCNNTransformer):
    def _build_optimizer(self, **kwargs):
        return Adam(lr=kwargs['lr'])

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _create_callbacks(self, **kwargs):
        return callbacks(**kwargs)


class WordSCNN(WordSCNNTransformer):
    def _build_optimizer(self, **kwargs):
        return Adam(lr=kwargs['lr'])

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _create_callbacks(self, **kwargs):
        return callbacks(**kwargs)


class WordDPCNN(WordDPCNNTransformer):
    def _build_optimizer(self, **kwargs):
        return Adam(lr=kwargs['lr'])

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _create_callbacks(self, **kwargs):
        return callbacks(**kwargs)


class WordCuDNNLSTM(WordCuDNNLSTMTransformer):
    def _build_optimizer(self, **kwargs):
        return Adam(lr=kwargs['lr'])

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _create_callbacks(self, **kwargs):
        return callbacks(**kwargs)


class WordCuDNNGRU(WordCuDNNGRUTransformer):
    def _build_optimizer(self, **kwargs):
        return Adam(lr=kwargs['lr'])

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _create_callbacks(self, **kwargs):
        return callbacks(**kwargs)


class StackerGru(ClassifierXY):
    def _build_optimizer(self, **kwargs):
        return Adam(lr=kwargs['lr'])

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _create_callbacks(self, **kwargs):
        return callbacks(**kwargs)

    def _build_model(self, unit_nr, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, rnn_dropout, dense_dropout, dropout_mode,
                     rnn_kernel_reg_l2, rnn_recurrent_reg_l2, rnn_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first):
        input_predictions = Input(shape=(output_size, 16))

        x = dropout_block(dropout_embedding, dropout_mode)(input_predictions)

        for _ in range(repeat_block):
            x = cudnn_gru_block(unit_nr=unit_nr, return_sequences=True, bidirectional=True,
                                kernel_reg_l2=rnn_kernel_reg_l2,
                                recurrent_reg_l2=rnn_recurrent_reg_l2,
                                bias_reg_l2=rnn_bias_reg_l2,
                                use_batch_norm=use_batch_norm, batch_norm_first=batch_norm_first,
                                dropout=rnn_dropout, dropout_mode=dropout_mode, use_prelu=use_prelu)(x)

        predictions = classification_block(dense_size=dense_size, repeat_dense=repeat_dense,
                                           output_size=output_size, output_activation=output_activation,
                                           max_pooling=max_pooling,
                                           mean_pooling=mean_pooling,
                                           weighted_average_attention=weighted_average_attention,
                                           concat_mode=concat_mode,
                                           dropout=dense_dropout,
                                           kernel_reg_l2=dense_kernel_reg_l2, bias_reg_l2=dense_bias_reg_l2,
                                           use_prelu=use_prelu, use_batch_norm=use_batch_norm,
                                           batch_norm_first=batch_norm_first)(x)
        model = Model(inputs=input_predictions, outputs=predictions)
        return model
