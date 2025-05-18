```python
|2025-05-16 21:42:58| {
     'ablation': 0, 'att_method': self, 'bs': 128
     'classification': False, 'continue_train': False, 'dataset': financial
     'debug': False, 'decay': 0.0001, 'density': 0.7
     'device': cpu, 'dis_method': cosine, 'end_date': 2025-05-15
     'epochs': 200, 'eval_set': True, 'ffn_method': ffn
     'fft': False, 'hyper_search': False, 'idx': 0
     'log': <utils.exp_logger.Logger object at 0x3644e4590>, 'logger': None, 'loss_func': MSELoss
     'lr': 0.001, 'model': ours, 'monitor_metric': MAE
     'multi_dataset': True, 'norm_method': rms, 'num_layers': 1
     'optim': AdamW, 'path': ./datasets/, 'patience': 45
     'pred_len': 5, 'rank': 56, 'record': True
     'retrain': True, 'revin': False, 'rounds': 1
     'scaler_method': stander, 'seed': 0, 'seq_len': 12
     'shuffle': False, 'start_date': 2020-05-15, 'train_size': 500
     'try_exp': 1, 'ts_var': 0, 'use_train_size': False
     'verbose': 5
}
|2025-05-16 21:42:58| ********************Experiment Start********************
|2025-05-16 21:48:29| Round=1 BestEpoch= 92 ValidMAE=0.0169 ｜ MAE=0.0145 RMSE=0.0204 NMAE=0.0105 NRMSE=0.0146 Acc_10=1.0000 time=199.1 s
|2025-05-16 21:48:29| ********************Experiment Results:********************
|2025-05-16 21:48:29| Dataset : FINANCIAL, Model : ours, Density : 0.700, Bs : 128, Rank : 56, Fundidx : 0, Seq_len : 12, Pred_len : 5, 
|2025-05-16 21:48:29| Train_length : 476 Valid_length : 54 Test_length : 125
|2025-05-16 21:48:29| MAE: 0.0145 ± 0.0000
|2025-05-16 21:48:29| RMSE: 0.0204 ± 0.0000
|2025-05-16 21:48:29| NMAE: 0.0105 ± 0.0000
|2025-05-16 21:48:29| NRMSE: 0.0146 ± 0.0000
|2025-05-16 21:48:29| Acc_10: 1.0000 ± 0.0000
|2025-05-16 21:48:29| train_time: 199.0610 ± 0.0000
|2025-05-16 21:49:14| Flops: 1862310912
|2025-05-16 21:49:14| Params: 25926
|2025-05-16 21:49:14| Inference time: 239.75 ms
|2025-05-16 21:49:14| ********************Experiment Success********************

|2025-05-16 21:49:14| Model(
  (model): Backbone(
    (projection): Linear(in_features=3, out_features=56, bias=True)
    (position_embedding): PositionEncoding(
      (pos_encoding): BertEmbedding(
        (embedding): Embedding(13, 56)
      )
    )
    (fund_embedding): Embedding(999999, 56)
    (predict_linear): Linear(in_features=12, out_features=17, bias=True)
    (encoder): Transformer(
      (layers): ModuleList(
        (0): ModuleList(
          (0): RMSNorm((56,), eps=None, elementwise_affine=True)
          (1): Attention(
            (att): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=56, out_features=56, bias=True)
            )
          )
          (2): RMSNorm((56,), eps=None, elementwise_affine=True)
          (3): FeedForward(
            (net): Sequential(
              (0): Linear(in_features=56, out_features=112, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=112, out_features=56, bias=True)
              (3): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
      (norm): RMSNorm((56,), eps=None, elementwise_affine=True)
    )
    (encoder2): Transformer(
      (layers): ModuleList(
        (0): ModuleList(
          (0): RMSNorm((56,), eps=None, elementwise_affine=True)
          (1): Attention(
            (att): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=56, out_features=56, bias=True)
            )
          )
          (2): RMSNorm((56,), eps=None, elementwise_affine=True)
          (3): FeedForward(
            (net): Sequential(
              (0): Linear(in_features=56, out_features=112, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=112, out_features=56, bias=True)
              (3): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
      (norm): RMSNorm((56,), eps=None, elementwise_affine=True)
    )
    (decoder): Linear(in_features=56, out_features=1, bias=True)
  )
  (distance): PairwiseLoss()
  (loss_function): MSELoss()
)
```
