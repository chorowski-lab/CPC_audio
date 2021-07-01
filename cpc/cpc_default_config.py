# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse


def get_default_cpc_config():
    parser = set_default_cpc_config(argparse.ArgumentParser())
    return parser.parse_args([])


def set_default_cpc_config(parser):
    # Run parameters

    group = parser.add_argument_group('Architecture configuration',
                                      description="The arguments defining the "
                                      "model's architecture.")
    group.add_argument('--hiddenEncoder', type=int, default=256,
                       help='Hidden dimension of the encoder network.')
    group.add_argument('--hiddenGar', type=int, default=256,
                       help='Hidden dimension of the auto-regressive network')
    group.add_argument('--nPredicts', type=int, default=12,
                       help='Number of steps to predict.')
    
    group.add_argument('--CPCCTC', action='store_true')
    group.add_argument('--CPCCTCNumMatched', type=int, default=16)
    group.add_argument('--CPCCTCSkipBeg', type=int, default=0)
    group.add_argument('--CPCCTCSkipEnd', type=int, default=0)
    group.add_argument('--CPCCTCSelfLoop', action='store_true')
    group.add_argument('--CPCCTCLearnBlank', action='store_true')
    group.add_argument('--CPCCTCNoNegsMatchWin', action='store_true')
    group.add_argument('--CPCCTCMasq', default="")
    group.add_argument('--CPCCTCLossTemp', type=float, default=1.0)
    group.add_argument('--CPCCTCNormalizeEncs', action='store_true')
    group.add_argument('--CPCCTCNormalizePreds', action='store_true')
    group.add_argument('--limitNegsInBatch', type=int, default=0,
                       help='Limit the number of different seqs from whithc neg samples are taken.')

    group.add_argument('--smartpoolingLayer', type=int, default=4, 
                       help='Which layers of the encoder should be replaced with smartpooling. Available layers: 3, 4, 5 (smart averaging)')
    group.add_argument('--smartpoolingNoPadding', action='store_true', 
                       help='No padding is added to encoder conv layer')
    group.add_argument('--smartpoolingDimMlp', type=int, default=2048, 
                       help='Dimension of the mlp responsible for assigning importance to frames.')
    group.add_argument('--smartpoolingUseDifferences', action='store_true',
                       help='Whether to not use mlp for importance and use abs of differences of consecutive values')                   
    group.add_argument('--smartpoolingTemperature', type=float, default=1e-5, 
                       help='Temperature added to frame importance. Larger temperature means the importance is going to be smoother')   
    group.add_argument('--smartaveragingWindowSize', type=int, default=None, 
                       help='How large the smart averaging window should be') 
    group.add_argument('--smartaveragingHardcodedWeights', action='store_true', 
                       help='Make the MLP output some hardcoded averaging weights')
    group.add_argument('--smartaveragingHardcodedWindowSize', type=int, default=None, 
                       help='How large the smart averaging HARDCODED window should be') 

    group.add_argument('--smartpoolingInAR', action='store_true',
                       help='Put smart averaging in AR. So archtecture is encoder -> (smart averaging -> AR) instead of (encoder -> smart averaging) -> AR') 
    group.add_argument('--smartpoolingInARUnfreezeEpoch', type=int, default=None, 
                       help='Which epoch to unfreeze the smartpooling in the AR. 0 means it is unfrozen from the start')
    group.add_argument('--smartaveragingLossParameter', type=float, default=None, 
                       help='The hyperparameter to scale the smart averaging loss. None means that the loss is not applied')
    group.add_argument('--smartaveragingLossAverage', type=float, default=None, 
                       help='Which value should the average aim towards')
    
    group.add_argument('--negativeSamplingExt', type=int, default=128,
                       help='Number of negative samples to take.')
    group.add_argument('--learningRate', type=float, default=2e-4)
    group.add_argument('--schedulerStep', type=int, default=-1,
                       help='Step of the learning rate scheduler: at each '
                       'step the learning rate is divided by 2. Default: '
                       'no scheduler.')
    group.add_argument('--schedulerRamp', type=int, default=None,
                       help='Enable a warm up phase for the learning rate: '
                       'adds a linear ramp of the given size.')
    group.add_argument('--beta1', type=float, default=0.9,
                       help='Value of beta1 for the Adam optimizer')
    group.add_argument('--beta2', type=float, default=0.999,
                       help='Value of beta2 for the Adam optimizer')
    group.add_argument('--epsilon', type=float, default=1e-08,
                       help='Value of epsilon for the Adam optimizer')
    group.add_argument('--sizeWindow', type=int, default=20480,
                       help='Number of frames to consider at each batch.')
    group.add_argument('--nEpoch', type=int, default=200,
                       help='Number of epoch to run')
    group.add_argument('--samplingType', type=str, default='samespeaker',
                       choices=['samespeaker', 'uniform',
                                'samesequence', 'sequential'],
                       help='How to sample the negative examples in the '
                       'CPC loss.')
    group.add_argument('--nLevelsPhone', type=int, default=1,
                       help='(Supervised mode only). Number of layers in '
                       'the phone classification network.')
    group.add_argument('--cpc_mode', type=str, default=None,
                       choices=['reverse', 'none'],
                       help='Some variations on CPC.')
    group.add_argument('--encoder_type', type=str,
                       choices=['cpc', 'mfcc', 'lfb', 'smart'],
                       default='cpc',
                       help='Replace the encoder network by mfcc features '
                       'or learned filter banks')
    group.add_argument('--normMode', type=str, default='layerNorm',
                       choices=['instanceNorm', 'ID', 'layerNorm',
                                'batchNorm'],
                       help="Type of normalization to use in the encoder "
                       "network (default is layerNorm).")
    group.add_argument('--onEncoder', action='store_true',
                       help="(Supervised mode only) Perform the "
                       "classification on the encoder's output.")
    group.add_argument('--random_seed', type=int, default=None,
                       help="Set a specific random seed.")
    group.add_argument('--speakerEmbedding', type=int, default=0,
                       help="(Depreciated) Feed the prediction network with "
                       "speaker embeddings along with the usual sequence.")
    group.add_argument('--arMode', default='LSTM',
                       choices=['GRU', 'LSTM', 'RNN', 'no_ar', 'transformer'],
                       help="Architecture to use for the auto-regressive "
                       "network (default is lstm).")
    group.add_argument('--nLevelsGRU', type=int, default=1,
                       help='Number of layers in the autoregressive network.')
    group.add_argument('--rnnMode', type=str, default='transformer',
                       choices=['transformer', 'RNN', 'LSTM', 'linear',
                                'ffd', 'conv4', 'conv8', 'conv12'],
                       help="Architecture to use for the prediction network")
    group.add_argument('--dropout', action='store_true',
                       help="Add a dropout layer at the output of the "
                       "prediction network.")
    group.add_argument('--abspos', action='store_true',
                       help='If the prediction network is a transformer, '
                       'active to use absolute coordinates.')

    return parser
