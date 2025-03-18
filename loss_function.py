import torch
import torch.nn as nn
import torch.nn.functional as F

from ADDvisor import audioprocessor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
#######################################
sablon pt a intelege acest loss combinat :) 
########################################





 def compute_objectives(self, pred, batch, stage):
        
        (
            batch_sig,
            predictions,
            xhat,
            _,
        ) = pred
        
        ########
        ## aici nu ma intereseaza, eu nu aplic augmentare din WHAM pt a avea nevoie de waveform "clean"
        ########
        batch = batch.to(self.device)
        wavs_clean, _ = batch.sig

        # taking them from forward because they are augmented there!
        wavs, _ = batch_sig
        #########
        ## label-uri encoded , again nu ma intereseaza acest aspect
        ########
        uttid = batch.id
        labels, _ = batch.class_string_encoded
        (
            ######################
            ### eu oricum fac compute la stft clean, deci iar, e in plus pt mine
            #####################
            
            X_stft_logpower_clean,
            _,
            _,
            _,
        ) = self.preprocess(wavs_clean)
        
        ##########
        ### primul param pt loss e stft log power...
        #######
        X_stft_logpower, _, _, _ = self.preprocess(wavs)

        Tmax = xhat.shape[1]
        
        ##################
        #### again, in plus
        ################
        # map clean to same dimensionality
        X_stft_logpower_clean = X_stft_logpower_clean[:, :Tmax, :]
        
        ##########
        #### cei mai importanti parametrii, masca de intrare si masca de iesire , practic sursa de la interpretabilitate....
        ########
        
        mask_in = xhat * X_stft_logpower[:, :Tmax, :]
        mask_out = (1 - xhat) * X_stft_logpower[:, :Tmax, :]
        
        ##############################
        #### nu folosesc mel spectrograma pt clasificare, deci skip 
        ###########################################################
        
        
        if self.hparams.use_stft2mel:
            X_in = torch.expm1(mask_in)
            mask_in_mel = self.hparams.compute_fbank(X_in)
            mask_in_mel = torch.log1p(mask_in_mel)

            X_out = torch.expm1(mask_out)
            mask_out_mel = self.hparams.compute_fbank(X_out)
            mask_out_mel = torch.log1p(mask_out_mel)
        
        ##########################
        # nu facem finetuning, skip
        #######################
        if self.hparams.finetuning:
            crosscor = self.crosscor(X_stft_logpower_clean, mask_in)
            crosscor_mask = (crosscor >= self.hparams.crosscor_th).float()

            max_batch = (
                X_stft_logpower_clean.view(X_stft_logpower_clean.shape[0], -1)
                .max(1)
                .values.view(-1, 1, 1)
            )
            binarized_oracle = (
                X_stft_logpower_clean >= self.hparams.bin_th * max_batch
            ).float()

            if self.hparams.guidelosstype == "binary":
                rec_loss = (
                    F.binary_cross_entropy(
                        xhat, binarized_oracle, reduce=False
                    ).mean((-1, -2))
                    * self.hparams.g_w
                    * crosscor_mask
                ).mean()
            else:
                temp = (
                    (
                        (
                            xhat
                            * X_stft_logpower[
                                :, : X_stft_logpower_clean.shape[1], :
                            ]
                        )
                        - X_stft_logpower_clean
                    )
                    .pow(2)
                    .mean((-1, -2))
                )
                rec_loss = (temp * crosscor_mask).mean() * self.hparams.g_w

        else:
            rec_loss = 0
            crosscor_mask = torch.zeros(xhat.shape[0], device=self.device)
        
        
        ##################################
        ######## predictiile clasificatorului pe cele doua spectrograme (practic spectrograma doar cu partile importante si cealalta parte a spectrogramei( fix opusul) 
        #####################################
        mask_in_preds = self.classifier_forward(mask_in_mel)[2]
        mask_out_preds = self.classifier_forward(mask_out_mel)[2]
        
        #################################
        ### predictia classifierului la inceput, yhat1
        ########################################
    
        class_pred = predictions.argmax(1)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #### loss cu spectrograma cu partile "importante" (mask in loss)
        l_in = F.nll_loss(mask_in_preds.log_softmax(1), class_pred)
        #### loss cu spectrograma cu partile irelevante ( mask out loss ) 
        l_out = -F.nll_loss(mask_out_preds.log_softmax(1), class_pred)
        #combines the 2 above losses with the weights 
        ao_loss = l_in * self.hparams.l_in_w + self.hparams.l_out_w * l_out
        
        # applies L1 regularization to mask values to ensure that the mask is going towards a binary mask ( 0 and 1 values ) 
        r_m = (
            xhat.abs().mean((-1, -2, -3))
            * self.hparams.reg_w_l1
            * torch.logical_not(crosscor_mask)
        ).sum()
        # total variation loss, ensures mask is smooth (quenstionable, idk if we need it )
        r_m += (
            tv_loss(xhat)
            * self.hparams.reg_w_tv
            * torch.logical_not(crosscor_mask)
        ).sum()

        mask_in_preds = mask_in_preds.softmax(1)
        mask_out_preds = mask_out_preds.softmax(1)

        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            self.inp_fid.append(
                uttid,
                mask_in_preds,
                predictions.softmax(1),
            )
            self.AD.append(
                uttid,
                mask_in_preds,
                predictions.softmax(1),
            )
            self.AI.append(
                uttid,
                mask_in_preds,
                predictions.softmax(1),
            )
            self.AG.append(
                uttid,
                mask_in_preds,
                predictions.softmax(1),
            )
            self.sps.append(uttid, wavs, X_stft_logpower, labels)
            self.comp.append(uttid, wavs, X_stft_logpower, labels)
            self.faithfulness.append(
                uttid,
                predictions.softmax(1),
                mask_out_preds,
            )

        self.in_masks.append(uttid, c=crosscor_mask)
        self.acc_metric.append(
            uttid,
            predict=predictions,
            target=labels,
        )

        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        return ao_loss + r_m + rec_loss
        
        
        
        def tv_loss(mask, tv_weight=1, power=2, border_penalty=0.3):
    if tv_weight is None or tv_weight == 0:
        return 0.0
    # https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
    # https://github.com/PiotrDabkowski/pytorch-saliency/blob/bfd501ec7888dbb3727494d06c71449df1530196/sal/utils/mask.py#L5
    w_variance = torch.sum(torch.pow(mask[:, :, :-1] - mask[:, :, 1:], power))
    h_variance = torch.sum(torch.pow(mask[:, :-1, :] - mask[:, 1:, :], power))

    loss = tv_weight * (h_variance + w_variance) / float(power * mask.size(0))
    return loss

"""


from audioprocessor import AudioProcessor

audio_processor = AudioProcessor()

class LMACLoss():
    def __init__(self, l_in_w=4.0, l_out_w=0.2, reg_w_l1=0.4, reg_w_tv=0.00, g_w=4.0):

        self.l_in_w = l_in_w #default =4.0
        self.l_out_w = l_out_w #default 0.2
        self.reg_w_l1 = reg_w_l1 #default 0.4
        self.reg_w_tv = reg_w_tv #default 0.00
        #self.g_w = g_w #default 4.0
        # these are for finetuning, leaving them out for now
        #self.crosscor_th = crosscor_th (0.6)
        #self.bin_th = bin_th (0.35)
        #self.guidelosstype = guidelosstype (binary)

    def loss_function(self, xhat, X_stft_power,X_stft_phase, class_pred, classifier_forward):


        # compute the masked spectrograms
        Tmax = xhat.shape[1]
        mask_in_mag = xhat * X_stft_power[:, :Tmax, :]  # the relevant parts of the spectrogram
        mask_out_mag = (1 - xhat) * X_stft_power[:, :Tmax, :] # the irelevant parts of the spectrogram

        # run the classifier using both masked_in and masked_out spectrograms
        # NEED TO COMPUTE ISTFT AND THEN REEXTRACT FEATS
        #print(mask_in.shape, mask_out.shape)
        #print(f"mask_in shape before ISTFT: {mask_in.shape}")
        mask_in = mask_in_mag * torch.exp(1j * X_stft_phase[:, :Tmax, :])
        mask_out = mask_out_mag * torch.exp(1j * X_stft_phase[:, :Tmax, :])

        istft_mask_in = audio_processor.compute_invert_stft(mask_in)
        istft_mask_out = audio_processor.compute_invert_stft(mask_out)
        # print(istft_mask_in.shape)
        # print(istft_mask_out.shape)
        istft_mask_in = istft_mask_in.squeeze(0)#.detach().cpu().numpy()
        istft_mask_out = istft_mask_out.squeeze(0)#.detach().cpu().numpy()
        #print(istft_mask_in.shape)
        #print(istft_mask_out.shape)
        mask_in_feats = audio_processor.extract_features_istft(istft_mask_in)
        mask_out_feats = audio_processor.extract_features_istft(istft_mask_out)
        mask_in_preds = classifier_forward(mask_in_feats)#[2]
        mask_out_preds = classifier_forward(mask_out_feats)#[2]
        print(mask_in_preds)
        print(mask_out_preds)
        mask_in_preds = torch.tensor(mask_in_preds).to(device)
        mask_out_preds = torch.tensor(mask_out_preds).to(device)

        # compute attribution objective loss (AO_loss)
        # negative log likelihood loss for both spectrograms after the mask
        # !!!!!!!!!! questionable, since this is usually used in multi-classification tasks !!!!!!!!!!!!!!!
        # l_in = F.nll_loss(mask_in_preds.log_softmax(1), class_pred)  #ensure that the classifier predicts the correct class using the relevant parts of the spectrogram
        # l_out = -F.nll_loss(mask_out_preds.log_softmax(1),
        #                     class_pred)  #ensure that the classifier DOES NOT predict the correct class using the irrelevant parts of the spectrogram

        # criterion = torch.nn.BCEWithLogitsLoss()
        #
        # l_in = criterion(mask_in_preds, class_pred)
        # l_out = -criterion(mask_out_preds, class_pred)

        l_in = F.binary_cross_entropy_with_logits(mask_in_preds, class_pred.float())
        l_out = -F.binary_cross_entropy_with_logits(mask_out_preds, class_pred.float())
        # print("xhat requires grad:", xhat.requires_grad)
        # print("X_stft_power requires grad:", X_stft_power.requires_grad)
        # print("X_stft_phase requires grad:", X_stft_phase.requires_grad)
        # print("mask_in_preds requires grad:", mask_in_preds.requires_grad)
        # print("mask_out_preds requires grad:", mask_out_preds.requires_grad)
        ao_loss = self.l_in_w * l_in + self.l_out_w * l_out

        # l1 regularization to force the mask to have values that are either 0 or 1
        reg_l1 = xhat.abs().mean((-1, -2)) * self.reg_w_l1

        # total variation loss, ensure the smoothness of the masks

        # tv_h = torch.sum(torch.abs(xhat[:, :, :-1] - xhat[:, :, 1:]))  # horizontal differences
        # tv_w = torch.sum(torch.abs(xhat[:, :-1, :] - xhat[:, 1:, :]))  # vertical differences
        # reg_tv = (tv_h + tv_w) * self.reg_w_tv


        #reg_loss = reg_l1 #+ reg_tv

        # compute correlation mask (only for finetuning)
        # if finetuning and X_stft_logpower_clean is not None:
        #     # cross-correlation mask computation
        #     crosscor = self.crosscor(X_stft_logpower_clean, mask_in)
        #     crosscor_mask = (crosscor >= self.crosscor_th).float()
        #
        #     # binarazied oracle
        #     max_batch = X_stft_logpower_clean.view(X_stft_logpower_clean.shape[0], -1).max(1).values.view(-1, 1, 1)
        #     binarized_oracle = (X_stft_logpower_clean >= self.bin_th * max_batch).float()
        #
        #     # compute reconstruction loss (I will leave it being only for the classification task, the regression is removed ... )
        #
        #     rec_loss = F.binary_cross_entropy(xhat, binarized_oracle, reduce=False).mean(
        #     (-1, -2)) * self.g_w * crosscor_mask
        #
        #
        #     rec_loss = rec_loss.mean()
        # else:
        #     rec_loss = 0
        #rec_loss = 0


        total_loss = ao_loss# + reg_loss + rec_loss
        return total_loss

    # def crosscor(self, spectrogram, template):
    #     """Compute cross-correlation metric."""
    #     #################################################
    #     ### identifies the relevant parts of the spectrogram and guides the mask during finetuning
    #     #################################################
    #     spectrogram = spectrogram - spectrogram.mean((-1, -2), keepdim=True)
    #     template = template - template.mean((-1, -2), keepdim=True)
    #
    #     dotp = (spectrogram * template).mean((-1, -2))
    #     norms_specs = spectrogram.pow(2).mean((-1, -2)).sqrt()
    #     norms_templates = template.pow(2).mean((-1, -2)).sqrt()
    #     return dotp / (norms_specs * norms_templates)
