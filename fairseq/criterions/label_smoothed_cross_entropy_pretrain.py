# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on
        parser.add_argument('--masked-lm-only', default=True,
                            help='compute MLM loss only')
        parser.add_argument('--nsp-loss-weight', default=1.0, type=float,
                            help='weight for next sentence prediction'
                                 ' loss (default 1)')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, lm_out = model(**sample['net_input'])
        lm_logits = lm_out[0]
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        lm_loss = self.compute_lm_loss(lm_logits, sample, reduce=reduce)
        logging_output = {
            'loss': utils.item(loss.data)+ utils.item(lm_loss.data) if reduce else loss.data + lm_loss.data,
            'nmt_loss': utils.item(loss.data) if reduce else loss.data,
            'lm_loss': utils.item(lm_loss.data)  if reduce else lm_loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        losses = loss + lm_loss
        return losses, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def compute_lm_loss(self, lm_logits, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        lm_logits = lm_logits.view(-1, lm_logits.size(-1))
        lm_targets = sample['target'].view(-1)
        lm_loss = compute_cross_entropy_loss(
            lm_logits, lm_targets, self.padding_idx)
        # compute the number of tokens for which loss is computed. This is used
        # to normalize the loss
        ntokens = utils.strip_pad(lm_targets, self.padding_idx).numel()
        
        '''
        if not self.args.masked_lm_only:
            sentence_logits = output_metadata['sentence_logits']
            sentence_targets = sample['sentence_target'].view(-1)
            # This needs to be recomputed due to some differences between
            # TokenBlock and BlockPair dataset. This can be resolved with a
            # refactor of BERTModel which we will do in the future.
            # TODO: Remove this after refactor of BERTModel
            nsentences = sentence_targets.size(0)

            # Check for logits being none which can happen when remove_heads
            # is set to true in the BERT model. Ideally we should set
            # masked_lm_only to true in this case, but that requires some
            # refactor in the BERT model.
            if sentence_logits is not None:
                sentence_loss = compute_cross_entropy_loss(
                    sentence_logits, sentence_targets)

                loss += self.args.nsp_loss_weight * (sentence_loss / nsentences)
        '''

        return lm_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nmt_loss': sum(log.get('nmt_loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'lm_loss': sum(log.get('lm_loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

def compute_cross_entropy_loss(logits, targets, ignore_index=-100):
    """
    Function to compute the cross entropy loss. The default value of
    ignore_index is the same as the default value for F.cross_entropy in
    pytorch.
    """
    assert logits.size(0) == targets.size(-1), \
        "Logits and Targets tensor shapes don't match up"
    
    logits = torch.add(logits, 1e-8)
    loss = F.cross_entropy(
        logits,
        targets,
        reduction="sum",
        ignore_index=ignore_index,
    )
    return loss
