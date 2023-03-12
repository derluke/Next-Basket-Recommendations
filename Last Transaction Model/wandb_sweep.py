
from pathlib import Path
import shutil
import wandb
import tensorflow as tf
import pandas as pd
import gc
from Configs import *
from Recommendation_Engine import Recommendation_Engine
from Autoencoder_Model import Autoencoder
from Model_Universe_Preparation import Model_Universe
from Model_Evaluation import Evaluator
from Configs import downsample_pct, cat_max_seq_length, cont_max_seq_length, categorical_cols, autoencoder_batch_size, autoencoder_epochs, latent_features, recommendation_batch_size, recommendation_epochs, workers, wd, fileName, runType, runTest
import wandb
from wandb.keras import WandbCallback




def train_recommender():
    
    run = wandb.init(project="basket-predictor", group="Recommender")

    optimiser = wandb.config.optimiser
    dense_1 = wandb.config.dense_1
    dense_2 = wandb.config.dense_2
    dropout = wandb.config.dropout


    wd_model = Path(wd)/Path(run.id)
    wd_model.mkdir(exist_ok=True, parents=True)
    wd_model = str(wd_model) + "/"
    mdl_univ_object = Model_Universe(wd = wd,\
                                     runType="model")
    model_universe, pur_seq, rtn_seq, tfidf_df, tf_idf_matrix = \
    mdl_univ_object.get_model_universe(call_all_functions=True,\
                                       fileName = fileName,\
                                       popularity_threshold=10,\
                                       customer_threshold=1,\
                                       last_n_txns=1,\
                                       downsample_pct=downsample_pct)
    gc.collect()
    if runTest:
        model_universe = model_universe[0:10000]
    encoder = tf.keras.models.load_model(wd + "keras-encoder.h5")

    shutil.copyfile(wd + "keras-encoder.h5", wd_model + "keras-encoder.h5")

    tf_idf_matrix = pd.DataFrame(data=encoder.predict(tf_idf_matrix.toarray()))
    gc.collect()

    
    recommendation_object = Recommendation_Engine(wd = wd_model,\
                                                  model_universe = model_universe,\
                                                  categorical_cols = categorical_cols,\
                                                  cat_max_seq_length = cat_max_seq_length,\
                                                  cont_max_seq_length = cont_max_seq_length,\
                                                  pur_seq = pur_seq,\
                                                  rtn_seq = rtn_seq,\
                                                  tf_idf_matrix = tf_idf_matrix,\
                                                  tfidf_df = tfidf_df,\
                                                  encoder = encoder,\
                                                  latent_features = latent_features,\
                                                  run_type="model")
    
    learn_ids, validation_ids = recommendation_object.preprocess_data()
    recommendation_object.create_model(dense_1=dense_1, dense_2=dense_2, optimiser=optimiser, dropout=dropout)
    print(recommendation_object.reco_model.summary())
    recommendation_object.fit_model(batch_size=recommendation_batch_size,\
                                    epochs=recommendation_epochs,\
                                    workers=workers,\
                                    shuffle=True,
                                    callbacks=[WandbCallback()],
                                    save_model=True)
    gc.collect()
    
    # Evaluation    
    eval_object = Evaluator(wd = wd_model,\
                            sample_ids_dict = {'Learn': learn_ids, 'Validation': validation_ids},\
                            model_universe = model_universe,\
                            pur_seq = pur_seq,\
                            rtn_seq = rtn_seq,\
                            tf_idf_matrix = tf_idf_matrix,\
                            tfidf_df = tfidf_df,\
                            batch_size=1024)
    
    eval_object.get_predictions(dense_1, dense_2, dropout, optimiser)
    hit_rates = eval_object.call_evaluation()
    
    for df, df_hrs in hit_rates.items():
        for top_n, hit_rate in df_hrs.items():
            wandb.log({
                f'{df}_hit_rate@{top_n}': hit_rate,
            })

sweep_config = {
        'method': 'random'
    }

metric = {
    'name': 'Validation_hit_rate@5',
    'goal': 'maximize'   
    }

sweep_config['metric'] = metric

parameters_dict = {
    'optimiser': {
        'values': ['adam', 'sgd']
        },
    'dense_1': {
        'values': [25, 50, 100]
        },
    'dense_2': {
        'values': [25, 50, 100]
    },
    'dropout': {
        'values': [True, False]
        },
}

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep=sweep_config, project='basket-predictor')

wandb.agent(sweep_id, function=train_recommender, count=10)
# train_recommender()