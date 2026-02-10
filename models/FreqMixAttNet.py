import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp,wavelet_series_decomp
from layers.Embed1 import DataEmbedding_wo_pos, Gating_DataEmbedding
from layers.StandardNorm import Normalize
import pytorch_wavelets as ptw
import pywt
from pytorch_wavelets import DWTForward
from layers.SelfAttention_Family import FullAttention, AttentionLayer, TSMixer, ResAttention


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend

class FFT_convolution(nn.Module):
    def __init__(self, embed_size, scale=0.03):
        super(FFT_convolution, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = scale
        
        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size)) # 1*T        

    def circular_convolution(self, x):
        w = self.w.to(x.device)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out, w
 
class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs, scale=0.02):
        super(MultiScaleSeasonMixing, self).__init__()
        self.scale = scale
        self.down_pool = torch.nn.AvgPool1d(configs.down_sampling_window)


        
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    )                    
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

        self.sr_ratio = configs.sr_ratio
       
        self.down_sampling_layers1 = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i)),
                    )                    
                )
                for i in range(configs.down_sampling_layers+1)
            ]
        )


        self.weights = [
                nn.Parameter(self.scale * torch.randn(1, configs.seq_len // (configs.down_sampling_window ** i)//2 +1)) 
                for i in range(configs.down_sampling_layers +1)
            ] 
        # print('------------------self.weights paramters in MultiScaleSeasonMixing---------------------')
        # for i in self.weights: print(i.shape )
        self.embed_size = [configs.seq_len // (configs.down_sampling_window ** i) for i in range(configs.down_sampling_layers +1 ) ]
                  
        self.d_model = configs.d_model
        self.weight_att = configs.weight_att
        nhead = configs.n_heads
        dropout=0.2

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )


        self.cross_attention1 =  TSMixer( self.cross_attention,self.d_model,nhead)
        self.patch_size = [24,24,24]
        
    def mix_att(self,i,season,sr_ratio,trend_list):
    
        #  seaon usring mlp
        out_low_res = self.down_sampling_layers1[i](season)
           

        out_season_list_t = out_low_res.permute(0, 2, 1)

        #season  using fft
        out_ = season
        # print('out_: ', out_.shape,'season: ', season.shape,out_season_list_t.shape)
        

        try:
            fft_compiled = torch.compile(
                torch.fft.rfft,
                mode="reduce-overhead",
                dynamic=True
            )
            out_ = fft_compiled(out_, dim=-1)
        except Exception as e:
            out_ = torch.fft.rfft(out_, dim=-1)

     
        w = self.weights[i].to(out_.device)
        out_ = out_ * w
        out_ = torch.fft.irfft(out_, n=self.embed_size[i], dim=2, norm="ortho")   
        out_season_list_f = out_.float().permute(0, 2, 1)


        #attention
        x_t = out_season_list_t
        x_f = out_season_list_f
        x_t1 = trend_list
        sr = sr_ratio
        B, L, D = x_f.size()
        x_f = x_f.reshape(B*sr , L//sr,D)
        x_t = x_t.reshape(B*sr , L//sr,D)
        x_t1 = x_t1.reshape(B*sr , L//sr,D)

        x3, attention_weights = self.cross_attention(
            query=x_t,
            key=x_f,
            value=x_f
        )
                        
        x3_1, attention_weights = self.cross_attention(
        query=x_t1,
        key=x_f,
        value=x_f
        )
        # print('x3: ', x3.shape,'x3_1: ', x3_1.shape)

        x3 = x3.reshape(B, L,D)
        x3_1 = x3_1.reshape(B, L,D)
        out_season_list = x3+ self.weight_att*x3_1

        return out_season_list
          

    def forward(self, season_list,trend_list):
        # mixing high->low
        out_season_list = []
        i  = 0
        sr_ratio = [4,2,1]
        if self.sr_ratio ==2:
            sr_ratio = [2,2,2]
        elif self.sr_ratio ==4:
            sr_ratio = [2,3,1]
        elif self.sr_ratio ==8:
            sr_ratio = [2,2,1]
        season = season_list[0]
        # print(i,"season:",season.shape)
        for i,sr,trend in zip(range(len(season_list)),sr_ratio,trend_list ):

                out_season_list_final = self.mix_att(i,season,sr,trend)
                # print ("out_season_list_final:",out_season_list_final.shape) 
                out_season_list.append(out_season_list_final)
                season  = self.down_pool(out_season_list_final.permute(0, 2, 1))
                # print(i,"season:",season.shape,out_season_list_final.shape)
          

        return out_season_list



class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = wavelet_series_decomp(configs.levels)
            # wavelet_series_decomp(configs.levels)
            # series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0:
            if configs.is_dff:
                self.cross_layer = nn.Sequential(
                    nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                    nn.GELU(),
                    nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
                )
            else:
                self.cross_layer = nn.Linear(in_features=configs.d_model, out_features=configs.d_model)

                # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs) 
        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)



        if configs.is_dff:
            self.out_cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )
        else:
            self.out_cross_layer = nn.Linear(in_features=configs.d_model, out_features=configs.d_model)
            

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            trend = self.decompsition(x.permute(0, 2, 1))
            season = x-trend
            # season, trend = self.decompsition(x)
            # print("x:",x.shape,"season",season.shape,"trend",trend.shape)
            # season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
            # trend_list.append(trend)


        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list,out_trend_list)
        out_list = []
        out_list_trend = []
        out_list_season  = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            # print("out:",out.shape,"season",out_season.shape,"trend",out_trend.shape,length)

            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
            out_list_season.append(out_season[:, :length, :])
            out_list_trend.append(out_trend[:, :length, :])
        return out_list,out_list_trend,out_list_season


class Model(nn.Module):

    def __init__(self, configs, use_future_temporal_feature=True):
        super(Model, self).__init__()
        if configs.is_seed:
            torch.manual_seed(configs.seed)
        self.configs = configs
        self.task_name = configs.task_name
        self.patch_len = configs.patch_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        print('configs.e_layers17:',configs.e_layers)
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = use_future_temporal_feature

        if self.channel_independence == 1:
            self.enc_embedding = Gating_DataEmbedding(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout,configs.patch_len)


            # self.enc_embedding = torch.nn.ModuleList(
            # [
            #             Gating_DataEmbedding(1, configs.d_model, configs.embed, configs.freq,configs.dropout,configs.patch_len)             
            #             for patch_len in [16,16,12]
            # ]
            # )
        else:
            self.enc_embedding = Gating_DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.normalize_layers0 = Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
        # num_hid_seq = 0
        # for i in range(configs.down_sampling_layers + 1):
        #     num_hid_seq += configs.seq_len // (configs.down_sampling_window ** i)
        # self.aug_predictor = torch.nn.Linear( num_hid_seq, configs.seq_len)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
            self.predict_layers_trend = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
            self.predict_layers_season = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
            
        # self.out_predictor = nn.Linear(configs.pred_len * 2, configs.pred_len)

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        # flg=0
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                # print('flag:',flg,x.shape, x_1.shape, x_2.shape)
                # flg+=1
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = [] #创建多尺度list
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1)) #加入原始粒度的尺度数据
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers): #加入粗粒度的尺度数据
            
            x_enc_sampling = down_pool(x_enc_ori)
            # print('down sample:' , x_enc_ori.shape, x_enc_sampling.shape)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, is_aug):
        # print(f"Available GPUs: {torch.cuda.device_count()}")
        # print(f"Current device: {torch.cuda.current_device()}")
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
        # print('before mul00:',x_enc.shape, x_mark_enc.shape)
        # # x_enc = self.enc_embedding(x_enc, None)  # [B,T,C]

        x = self.normalize_layers0(x_enc, 'norm')
        if self.channel_independence == 1:
            Batch_size, T, N = x.size()
            x_enc = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_mark_enc = x_mark_enc.repeat(N, 1, 1)

        # x_mark_dec = self.enc_embedding(None, x_mark_dec)
            # print('before mul0:',x_enc.shape, x_mark_enc.shape,self.x_mark_dec.shape )
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)


        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                # print('atfer mul:',x.shape, x_mark.shape)
                # print('before mul:',x, x_mark)
                B, T, N = x.size()
                # x = self.normalize_layers[i](x, 'norm')
                # if self.channel_independence == 1:
                #     x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                #     x_mark = x_mark.repeat(N, 1, 1)
                # print('before mul1:',x.shape, x_mark.shape)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        # print('#'*100)
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                # print(i, 'after preenc',x.shape, x_mark.shape)
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                # print(i, 'after preenc1',enc_out.shape)
                # enc_out= x + x_mark  * self.weight
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list,out_list_trend,out_list_season = self.pdm_blocks[i](enc_out_list)
        B = Batch_size
        # print('enc_out_list shape:',len(enc_out_list),enc_out_list[0].shape, enc_out_list[1].shape, enc_out_list[-1].shape)
        if is_aug:
            dec_out_list, dec_out_list1 = self.future_multi_mixing(B, enc_out_list, x_list,self.predict_layers)

            dec_out1 = torch.concat(dec_out_list1, dim=1)
            # print('dec_out1 shape:',dec_out_list1.shape)

            dec_out = torch.stack(dec_out_list, dim=-1)
            dec_out = dec_out.sum(-1)
            # print('dec_out3 shape:',dec_out.shape)
            # dec_out1 = self.normalize_layers[0](dec_out1, 'denorm')            
            # dec_out = self.normalize_layers[0](dec_out, 'denorm')            
            dec_out = self.normalize_layers0(dec_out, 'denorm')        



            dec_out_list, dec_out_list1 = self.future_multi_mixing(B, out_list_trend, x_list,self.predict_layers_trend)    
            dec_out_trend = torch.stack(dec_out_list, dim=-1)
            dec_out_trend = dec_out_trend.sum(-1)
            dec_out_trend = self.normalize_layers0(dec_out_trend, 'denorm')   
            
            dec_out_list, dec_out_list1 = self.future_multi_mixing(B, out_list_season, x_list,self.predict_layers_season)  
            dec_out_season = torch.stack(dec_out_list, dim=-1)
            dec_out_season = dec_out_season.sum(-1)
            dec_out_season = self.normalize_layers0(dec_out_season, 'denorm')  

            return dec_out1, dec_out,dec_out_trend ,dec_out_season
        else:
            dec_out_list, _ = self.future_multi_mixing(B, enc_out_list, x_list,self.predict_layers)

            dec_out = torch.stack(dec_out_list, dim=-1)

            dec_out = dec_out.sum(-1)
            
            # dec_out = self.out_predictor(torch.concat([dec_out, dec_out], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
            dec_out = self.normalize_layers0(dec_out, 'denorm')
            return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list,predict_layer):
        dec_out_list = []
        dec_out_list1 = []
        pred_len = [96,48,24]
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out,pl in zip(range(len(x_list)), enc_out_list,pred_len):
                # dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1))  # align temporal dimension

                dec_out = predict_layer[i](enc_out.permute(0, 2, 1))  # align temporal dimension
                # dec_out_list1.append(dec_out.reshape(B, -1, self.pred_len).permute(0 , 2, 1))

                dec_out = dec_out.permute(0 , 2, 1)
                # print("dec_out:",dec_out.shape,enc_out.reshape(B, pl,-1).shape,enc_out.shape)
                dec_out_list1.append(enc_out.reshape(B, pl,-1))
                # dec_outii= dec_out.reshape(B, self.pred_len,-1)
                # print("dec_outii:",dec_outii.shape,dec_out.shape)
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                # print("dec_out1:",dec_out.shape,B,self.configs.c_out, self.pred_len)    
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1))# align temporal dimension
                # dec_out_list1.append(dec_out.reshape(B, -1, self.pred_len).permute(0 , 2, 1))
                dec_out = dec_out.permute(0 , 2, 1)
                dec_out_list1.append(dec_out.reshape(B, self.pred_len,-1))
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list, dec_out_list1

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, is_aug=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # print("GCN x_enc:",x_enc.shape)
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, is_aug)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError('Other tasks implemented yet')
