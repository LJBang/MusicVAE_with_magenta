from magenta.models.music_vae import data, lstm_models
from magenta.models.music_vae.configs import Config, CONFIG_MAP, HParams
from magenta.models.music_vae.base_model import MusicVAE
from magenta.common import merge_hparams
from magenta.models.music_vae.music_vae_train import console_entry_point

CONFIG_MAP['flat-drums_4bar'] = Config(
    model=MusicVAE(encoder=lstm_models.BidirectionalLstmEncoder(),
                    decoder=lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=64,
            z_size=256,
            enc_rnn_size=[512, 512],
            dec_rnn_size=[256, 256],
            free_bits=48,
            max_beta=0.2,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000
        )
    ),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
        max_bars=100,
        slice_bars=4,
        steps_per_quarter=4,
        roll_input=True
    ),
    train_examples_path=None,
    eval_examples_path=None
)

CONFIG_MAP['hierdec-drums_4bar'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.HierarchicalLstmDecoder(
            lstm_models.CategoricalLstmDecoder(),
            level_lengths=[16, 4],
            disable_autoregression=True)),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=64,
            z_size=256,
            enc_rnn_size=[512, 512],
            dec_rnn_size=[256, 256],
            free_bits=48,
            max_beta=0.2,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000
        )),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
        max_bars=100,
        slice_bars=4,
        steps_per_quarter=4,
        roll_input=True,
    ),
    train_examples_path=None,
    eval_examples_path=None,
)

if __name__ == '__main__':
    console_entry_point()