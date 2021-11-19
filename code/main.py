import os
import numpy as np
import random

import torch
import torch.nn as nn

from model import *

import arguments

import utils.load_dataset
import utils.data_loader
import utils.metrics
from utils.early_stop import EarlyStopping, Stop_args, StopVariable
import time
import matplotlib.pyplot as plt
import argparse
import fitlog
import debugpy
# debugpy.listen(('0.0.0.0',8888))
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def para(args): 
    if args.dataset == 'yahooR3': 
        args.training_args = {'batch_size': 1024, 'epochs': 40000, 'patience': 80, 'block_batch': [6000, 500]}
    elif args.dataset == 'coat':
        args.training_args = {'batch_size': 128, 'epochs': 40000, 'patience': 80, 'block_batch': [64, 64]}
    elif args.dataset == 'kuai':
        args.training_args = {'batch_size': 1024, 'epochs': 20000, 'patience': 80, 'block_batch': [6000, 500]}
    else: 
        print('invalid arguments')
        os._exit()

def gap(loader, CF_model, F_model, user_num):
    label_dict={}
    CF_dict={}
    F_dict={}
    CF_mask = torch.zeros(user_num)
    F_mask = torch.zeros(user_num)
    for batch_idx, (users, items, ratings) in enumerate(loader):
        CF_pred = CF_model(users, items)
        F_pred = F_model(users, items)
        for i,u in enumerate(users):
            try:
                label_dict[u.item()].append(ratings[i].item())
                CF_dict[u.item()].append(CF_pred[i].item())
                F_dict[u.item()].append(F_pred[i].item())
            except:
                label_dict[u.item()]=[ratings[i].item()]
                CF_dict[u.item()]=[CF_pred[i].item()]
                F_dict[u.item()]=[F_pred[i].item()]

    for u in list(label_dict.keys()): 
        if (-1.0 in label_dict[u]) and (1.0 in label_dict[u]):
            pos_mask = np.where(np.array(label_dict[u]) == 1.0)
            neg_mask = np.where(np.array(label_dict[u]) == -1.0)
            CF_gap_one = min(np.array(CF_dict[u])[pos_mask]) - max(np.array(CF_dict[u])[neg_mask])
            F_gap_one = min(np.array(F_dict[u])[pos_mask]) - max(np.array(F_dict[u])[neg_mask])
            if CF_gap_one >= F_gap_one:
                CF_mask[u] = 1
            else:
                F_mask[u] = 1
    return CF_mask, F_mask

def both_test(loader, model_name, testname, K = 5, dataset = "None"):
    if dataset == "kuai":
        test_results={}
        u_i_dict={}
        u_r_dict={}
        for batch_idx, (users, items, ratings) in enumerate(loader):
            for i,u in enumerate(users):
                if ratings[i]>=1:
                    GroundTruth = 1
                elif ratings[i]<=-1:
                    GroundTruth = -1
                else:
                    GroundTruth = ratings[i]
                try:
                    u_i_dict[u.item()].append(items[i].item())
                    u_r_dict[u.item()].append(GroundTruth)
                except:
                    u_i_dict[u.item()]=[items[i].item()]
                    u_r_dict[u.item()]=[GroundTruth]

        uauc_all = 0.0
        ndcg_all=[0.0, 0.0, 0.0, 0.0]
        test_user_num=0.0
        TEST_U=list(u_i_dict.keys())
        for k in TEST_U:
            if (1 in u_r_dict[k]) and (-1.0 in u_r_dict[k]):
                test_user_num += 1

                item = u_i_dict[k]
                label = torch.tensor(u_r_dict[k]).to(device)
                vector_predict = model_name(torch.tensor(k).to(device), torch.tensor(item).to(device))

                uauc_one = utils.metrics.auc(vector_predict, label)
                uauc_all = uauc_all + uauc_one

                for ki,ndcgK in enumerate([5, 20, 50, 100]):
                    atk=min(len(u_r_dict[k]), ndcgK)
                    hit, dcg, idcg = 0, 0, 0
                    rank=torch.topk(vector_predict, atk)[1]
                    count_1 = (len(u_r_dict[k]) + sum(u_r_dict[k]))/2
                    '''
                    assume that P is the number of positive examples and N is the number of negative examples
                    then len(u_r_dict[k])=P+N,sum(u_r_dict[k])=P*1+N*(-1)=P-N
                    P=(len(u_r_dict[k])+sum(u_r_dict[k]))/2
                    N=(len(u_r_dict[k])-sum(u_r_dict[k]))/2
                    then count_1=P=(len(u_r_dict[k])+sum(u_r_dict[k]))/2
                    '''
                    ideal=min(atk, count_1)
                    for j in range(atk):
                        if u_r_dict[k][rank[j]] ==1 :
                            hit, dcg = hit + 1, dcg + 1 / np.log2(j + 2)
                        if ideal>0:
                            idcg = idcg + 1 / np.log2(j + 2)
                            ideal = ideal-1
                    ndcg_all[ki] += dcg/idcg

        test_results['NDCG_5'] = ndcg_all[0]/test_user_num
        test_results['NDCG_20'] = ndcg_all[1]/test_user_num
        test_results['NDCG_50'] = ndcg_all[2]/test_user_num
        test_results['NDCG_100'] = ndcg_all[3]/test_user_num
        test_results['UAUC'] = uauc_all/test_user_num
            
        fitlog.add_best_metric({f"{testname[1]}_{testname[2]}_test":{f"NDCG_5":test_results['NDCG_5'], f"NDCG_20":test_results['NDCG_20'], f"NDCG_50":test_results['NDCG_50'], f"NDCG_100":test_results['NDCG_100'], f"UAUC":test_results['UAUC']}})
    else:
        test_users = torch.empty(0, dtype=torch.int64).to(device)
        test_items = torch.empty(0, dtype=torch.int64).to(device)
        test_pre_ratings = torch.empty(0).to(device)
        test_ratings = torch.empty(0).to(device)
        ndcg_ratings = torch.empty(0).to(device)
        ndcg_item = torch.empty(0).to(device)
        ut_dict={}
        pt_dict={}
        for batch_idx, (users, items, ratings) in enumerate(loader):
            pre_ratings = model_name(users, items)
            for i,u in enumerate(users):
                try:
                    ut_dict[u.item()].append(ratings[i].item())
                    pt_dict[u.item()].append(pre_ratings[i].item())
                except:
                    ut_dict[u.item()]=[ratings[i].item()]
                    pt_dict[u.item()]=[pre_ratings[i].item()]
            test_users = torch.cat((test_users, users))
            test_items = torch.cat((test_items, items))
            test_pre_ratings = torch.cat((test_pre_ratings, pre_ratings))
            test_ratings = torch.cat((test_ratings, ratings))

            pos_mask = torch.where(ratings>=torch.ones_like(ratings), torch.arange(0,len(ratings)).float().to(device), 100*torch.ones_like(ratings))
            pos_ind = pos_mask[pos_mask != 100].long()
            users_ndcg = torch.index_select(users, 0, pos_ind)
            ratings_ndcg = model_name.allrank(users_ndcg, bias_train)
            ndcg_ratings = torch.cat((ndcg_ratings, ratings_ndcg))
            items = torch.index_select(items.float(), 0, pos_ind)
            ndcg_item= torch.cat((ndcg_item, items))

        test_results = utils.metrics.evaluate(test_pre_ratings, test_ratings, ['MSE', 'NLL', 'AUC', 'Recall_Precision_NDCG@'], users=test_users, items=test_items, NDCG=(ndcg_ratings, ndcg_item), UAUC=(ut_dict, pt_dict))
        print('The {} performance of {} testing set: {}'.format(testname[0], testname[2], ' '.join([key+':'+'%.3f'%test_results[key] for key in test_results])))
        fitlog.add_best_metric({f"{testname[1]}_{testname[2]}_test":{"MSE":test_results['MSE'], "NLL":test_results['NLL'], "AUC":test_results['AUC'], "UAUC":test_results['UAUC'], "NDCG":test_results['NDCG']}})
    return test_results

def step_test(loader, model_name, testname, epoch, dataset):
    if dataset=='kuai':
        try:
            test_results={}
            u_i_dict={}
            u_r_dict={}
            for batch_idx, (users, items, ratings) in enumerate(loader):
                for i,u in enumerate(users):
                    if ratings[i]>=1:
                        GroundTruth = 1
                    elif ratings[i]<=-1:
                        GroundTruth = -1
                    else:
                        GroundTruth = ratings[i]
                    try:
                        u_i_dict[u.item()].append(items[i].item())
                        u_r_dict[u.item()].append(GroundTruth)
                    except:
                        u_i_dict[u.item()]=[items[i].item()]
                        u_r_dict[u.item()]=[GroundTruth]

            uauc_all = 0.0
            ndcg_all=[0.0, 0.0, 0.0, 0.0]
            test_user_num=0.0
            TEST_U=list(u_i_dict.keys())
            for k in TEST_U:
                if (1 in u_r_dict[k]) and (-1.0 in u_r_dict[k]):
                    test_user_num += 1

                    item = u_i_dict[k]
                    label = torch.tensor(u_r_dict[k]).to(device)
                    vector_predict = model_name(torch.tensor(k).to(device), torch.tensor(item).to(device))

                    uauc_one = utils.metrics.auc(vector_predict, label)
                    uauc_all = uauc_all + uauc_one

                    for ki,ndcgK in enumerate([5, 20, 50, 100]):
                        atk=min(len(u_r_dict[k]), ndcgK)
                        hit, dcg, idcg = 0, 0, 0
                        rank=torch.topk(vector_predict, atk)[1]
                        count_1 = (len(u_r_dict[k]) + sum(u_r_dict[k]))/2
                        '''
                        assume that P is the number of positive examples and N is the number of negative examples
                        then len(u_r_dict[k])=P+N,sum(u_r_dict[k])=P*1+N*(-1)=P-N
                        P=(len(u_r_dict[k])+sum(u_r_dict[k]))/2
                        N=(len(u_r_dict[k])-sum(u_r_dict[k]))/2
                        then count_1=P=(len(u_r_dict[k])+sum(u_r_dict[k]))/2
                        '''
                        ideal=min(atk, count_1)
                        for j in range(atk):
                            if u_r_dict[k][rank[j]] ==1 :
                                hit, dcg = hit + 1, dcg + 1 / np.log2(j + 2)
                            if ideal>0:
                                idcg = idcg + 1 / np.log2(j + 2)
                                ideal = ideal-1
                        ndcg_all[ki] += dcg/idcg

            test_results['NDCG_5'] = ndcg_all[0]/test_user_num
            test_results['NDCG_20'] = ndcg_all[1]/test_user_num
            test_results['NDCG_50'] = ndcg_all[2]/test_user_num
            test_results['NDCG_100'] = ndcg_all[3]/test_user_num
            test_results['UAUC'] = uauc_all/test_user_num
            fitlog.add_metric({f"tes_{testname}":{"UAUC":test_results['UAUC']}}, step=epoch)
            fitlog.add_metric({f"tes_{testname}":{"NDCG5":test_results['NDCG_5']}}, step=epoch)
            fitlog.add_metric({f"tes_{testname}":{"NDCG20":test_results['NDCG_20']}}, step=epoch)
            fitlog.add_metric({f"tes_{testname}":{"NDCG50":test_results['NDCG_50']}}, step=epoch)
            fitlog.add_metric({f"tes_{testname}":{"NDCG100":test_results['NDCG_100']}}, step=epoch)
        except:
            dsh=1
    else:
        test_users = torch.empty(0, dtype=torch.int64).to(device)
        test_items = torch.empty(0, dtype=torch.int64).to(device)
        test_pre_ratings = torch.empty(0).to(device)
        test_ratings = torch.empty(0).to(device)
        ndcg_ratings = torch.empty(0).to(device)
        ndcg_item = torch.empty(0).to(device)
        ut_dict={}
        pt_dict={}
        for batch_idx, (users, items, ratings) in enumerate(loader):
            pre_ratings = model_name(users, items)
            for i,u in enumerate(users):
                try:
                    ut_dict[u.item()].append(ratings[i].item())
                    pt_dict[u.item()].append(pre_ratings[i].item())
                except:
                    ut_dict[u.item()]=[ratings[i].item()]
                    pt_dict[u.item()]=[pre_ratings[i].item()]
            test_users = torch.cat((test_users, users))
            test_items = torch.cat((test_items, items))
            test_pre_ratings = torch.cat((test_pre_ratings, pre_ratings))
            test_ratings = torch.cat((test_ratings, ratings))

            pos_mask = torch.where(ratings>=torch.ones_like(ratings), torch.arange(0,len(ratings)).float().to(device), 100*torch.ones_like(ratings))
            pos_ind = pos_mask[pos_mask != 100].long()
            users_ndcg = torch.index_select(users, 0, pos_ind)
            ratings_ndcg = model_name.allrank(users_ndcg, bias_train)
            ndcg_ratings = torch.cat((ndcg_ratings, ratings_ndcg))
            items = torch.index_select(items.float(), 0, pos_ind)
            ndcg_item= torch.cat((ndcg_item, items))
        test_results = utils.metrics.evaluate(test_pre_ratings, test_ratings, ['MSE', 'NLL', 'AUC', 'Recall_Precision_NDCG@'], users=test_users, items=test_items, NDCG=(ndcg_ratings, ndcg_item), UAUC=(ut_dict, pt_dict))
        fitlog.add_metric({f"tes_{testname}":{"MSE":test_results['MSE']}}, step=epoch)
        fitlog.add_metric({f"tes_{testname}":{"NLL":test_results['NLL']}}, step=epoch)
        fitlog.add_metric({f"tes_{testname}":{"AUC":test_results['AUC']}}, step=epoch)
        fitlog.add_metric({f"tes_{testname}":{"UAUC":test_results['UAUC']}}, step=epoch)
        fitlog.add_metric({f"tes_{testname}":{"NDCG":test_results['NDCG']}}, step=epoch)
    return test_results

def distribution_display(m, n,data, name, sort = None):
    # print(type(data))
    users = data._indices()[0].cpu().numpy()
    items = data._indices()[1].cpu().numpy()
    user_his = np.bincount(users, minlength=m)
    item_his = np.bincount(items, minlength=n)
    if sort == None:
        Index_user = np.argsort(user_his)
        Index_item = np.argsort(item_his)
    else:
        Index_user = sort[0]
        Index_item = sort[1]

    y1 = user_his[Index_user]
    x1 = range(len(y1))
    y2 = item_his[Index_item]
    x2 = range(len(y2))
    plt.figure()
    plt.bar(x1, y1, label=name+'_user')
    plt.legend()
    plt.savefig('./Dis_fig/'+ args.dataset+ '_'+ name+'_user.png')

    plt.figure()
    plt.bar(x2, y2, label=name+'_item')
    plt.legend()
    plt.savefig('./Dis_fig/'+ args.dataset+ '_'+ name+'_item')
    sort = (Index_user, Index_item)
    return sort

def train_and_eval_CFF(bias_train, bias_validation, bias_test, unif_validation, unif_test, m, n, device = 'cuda', 
        base_model_args: dict = {'emb_dim': 64, 'learning_rate': 0.05, 'imputaion_lambda': 0.01, 'weight_decay': 0.05}, 
        weight1_model_args: dict = {'learning_rate': 0.1, 'weight_decay': 0.005}, 
        weight2_model_args: dict = {'learning_rate': 0.1, 'weight_decay': 0.005}, 
        imputation_model_args: dict = {'learning_rate': 0.01, 'weight_decay': 0.5,'bias': 0}, 
        training_args: dict =  {'batch_size': 1024, 'epochs': 100, 'patience': 20, 'block_batch': [1000, 100]}, weightName = ' ', gama=1, args=None):
    
    train_dense = bias_train.to_dense()
    if args.dataset == 'coat' or args.dataset == 'kuai':
        train_dense_norm = torch.where(train_dense<-1*torch.ones_like(train_dense), -1*torch.ones_like(train_dense), train_dense)
        train_dense_norm = torch.where(train_dense_norm>torch.ones_like(train_dense_norm), torch.ones_like(train_dense_norm), train_dense_norm)
        del train_dense
        train_dense = train_dense_norm
    # uniform data
    # uniform data
    
    # build data_loader. (block matrix data loader)
    train_loader = utils.data_loader.Block(bias_train, u_batch_size=training_args['block_batch'][0], i_batch_size=training_args['block_batch'][1], device=device)
    biasval_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(bias_validation), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    biastest_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(bias_test), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)

    val_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(unif_validation), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(unif_test), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    # impu_dataloader = utils.data_loader.DataLoader(utils.data_loader.Interactions(bias_train), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    # unlabel_loader = utils.data_loader.NoLabelBlock(bias_train, times=1, device=device)
    # data shape
    # n_user, n_item = train_data.shape
    n_user, n_item = m, n
    
    # Base model and its optimizer. This optimizer is for optimize parameters in base model using the updated weights (true optimization).
    CFF_model = MF_MSE(n_user, n_item, dim=base_model_args['emb_dim'], dropout=0).to(device)
    base_optimizer = torch.optim.SGD(CFF_model.params(), lr=base_model_args['learning_rate'], weight_decay=0) # todo: other optimizer SGD
    CF_model = MetaMF(n_user, n_item, dim=base_model_args['emb_dim'], dropout=0).to(device)
    CF_model.load_state_dict(torch.load(f"/home/auto_{args.dataset}_{weightName}.pth.tar"))
    F_model = MF_MSE(n_user, n_item, dim=base_model_args['emb_dim'], dropout=0).to(device)
    F_model.load_state_dict(torch.load(f"/home/mf_{args.dataset}_{weightName}.pth.tar"))
    weight1_model = ThreeLinear(n_user, n_item, 2).to(device)
    weight1_model.load_state_dict(torch.load(f"/home/weight1_{args.dataset}_{weightName}.pth.tar"))
    weight2_model = ThreeLinear(n_user, n_item, 2).to(device)
    weight2_model.load_state_dict(torch.load(f"/home/weight2_{args.dataset}_{weightName}.pth.tar"))
    imputation_model = OneLinear(3).to(device)
    imputation_model.load_state_dict(torch.load(f"/home/imputation_{args.dataset}_{weightName}.pth.tar"))
    # G_model = Grad_model(n_user, n_item, dim=base_model_args['emb_dim'], dropout=0, Fmodel_weight = F_model, CFmodel_weight =CF_model).to(device)
    
    # loss_criterion
    sum_criterion = nn.MSELoss(reduction='sum')

    # begin training
    stopping_args = Stop_args(stop_varnames=[StopVariable.AUC, StopVariable.AUC2], patience=training_args['patience'], max_epochs=training_args['epochs'])
    early_stopping_cff = EarlyStopping(CFF_model, **stopping_args)
    fitlog.add_best_metric({"bias_val":{"Earlystop":0}})

    # CF_mask, F_mask = gap(impu_dataloader, CF_model, F_model, m)
    # CF_mask, F_mask = CF_mask.to(device), F_mask.to(device)

    for epo in range(training_args['epochs']):
        training_loss = 0
        lossf_sum = 0
        lossl_sum=0
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                users_train, items_train, y_train = train_loader.get_batch(users, items)
                if args.dataset == 'coat' or args.dataset == 'kuai':
                    y_train = torch.where(y_train<-1*torch.ones_like(y_train), -1*torch.ones_like(y_train), y_train)
                    y_train = torch.where(y_train >1*torch.ones_like(y_train), torch.ones_like(y_train), y_train)
                CF_pred = CF_model.forward(users_train, items_train)
                F_pred = F_model.forward(users_train, items_train)
                weight1 = weight1_model(users_train, items_train,(y_train==1)*1)
                weight1 = torch.exp(weight1/5)
                Auto_loss = nn.MSELoss(reduction='none')(CF_pred, y_train)

                # all pair
                all_pair = torch.cartesian_prod(users, items)
                users_all, items_all = all_pair[:,0], all_pair[:,1]
                values_all = train_dense[users_all, items_all]
                obs_mask = torch.abs(values_all)

                weight2 = weight2_model(users_train, items_train,(train_dense[users_train,items_train]!=0)*1)
                weight2 = torch.exp(weight2/5)
                impu_train = torch.tanh(imputation_model((train_dense[users_train,items_train]).long()+1))
                cost_impu = nn.MSELoss(reduction='none')(CF_pred, impu_train)

                CF_loss = Auto_loss* weight1 + cost_impu* weight2
                F_loss = nn.MSELoss(reduction='none')(F_pred, y_train)

				#lossA train
                users_no, items_no, values_all = users_all, items_all, train_dense[users_all, items_all]

                CF_pred_A = CF_model.forward(users_no, items_no)
                F_pred_A = F_model.forward(users_no, items_no)
                y_hat_obsA = CFF_model(users_no, items_no)
                Loss_FA = nn.MSELoss(reduction='none')(y_hat_obsA, F_pred_A)
                weight2A = weight2_model(users_no, items_no,(train_dense[users_no,items_no]!=0)*1)
                weight2A = torch.exp(weight2A/5)
                impu_trainA = torch.tanh(imputation_model((train_dense[users_no,items_no]).long()+1))
                Loss_CFA = nn.MSELoss(reduction='none')(CF_pred_A, y_hat_obsA)* weight2A

                W_CFA = torch.pow(Loss_FA, args.gama2) / (torch.pow(Loss_CFA, args.gama2) + torch.pow(Loss_FA, args.gama2))
                W_FA = torch.pow(Loss_CFA, args.gama2)  / (torch.pow(Loss_CFA, args.gama2) + torch.pow(Loss_FA, args.gama2))
                y_causal_trainA = W_CFA * CF_pred_A + W_FA * F_pred_A

                y_hat_obs_A = CFF_model(users_no, items_no)
                loss_A = nn.MSELoss(reduction='none')(y_hat_obs_A, y_causal_trainA)
                imp_mask = torch.ones_like(values_all)-torch.abs(values_all)
                loss_A = torch.sum(loss_A*imp_mask)

                #do causal fusion
                W_CF = torch.pow(F_loss, gama) / (torch.pow(CF_loss, gama) + torch.pow(F_loss, gama))
                W_F = torch.pow(CF_loss, gama)  / (torch.pow(CF_loss, gama) + torch.pow(F_loss, gama))
                y_causal_train = W_CF * CF_pred + W_F * F_pred

                # loss of training set
                CFF_model.train()
                # observation
                y_hat_obs = CFF_model(users_train, items_train)
                cost_obs = sum_criterion(y_hat_obs, y_causal_train)

                loss = cost_obs + args.beta*loss_A + base_model_args['weight_decay'] * CFF_model.l2_norm(users_all, items_all)

                base_optimizer.zero_grad()
                loss.backward()
                base_optimizer.step()

                training_loss += loss.item()


        CFF_model.eval()
        with torch.no_grad():
            # training metrics
            train_pre_ratings = torch.empty(0).to(device)
            train_ratings = torch.empty(0).to(device)
            for u_batch_idx, users in enumerate(train_loader.User_loader): 
                for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                    users_train, items_train, y_train= train_loader.get_batch(users, items)
                    pre_ratings = CFF_model(users_train, items_train)
                    train_pre_ratings = torch.cat((train_pre_ratings, pre_ratings))
                    train_ratings = torch.cat((train_ratings, y_train))

            train_results = utils.metrics.evaluate(train_pre_ratings, train_ratings, ['MSE', 'NLL'])

            # validation metrics on unifi
            un_val_pre_ratings = torch.empty(0).to(device)
            un_val_ratings = torch.empty(0).to(device)
            for batch_idx, (users, items, ratings) in enumerate(val_loader):
                pre_ratings = CFF_model(users, items)
                un_val_pre_ratings = torch.cat((un_val_pre_ratings, pre_ratings))
                un_val_ratings = torch.cat((un_val_ratings, ratings))

            un_val_results = utils.metrics.evaluate(un_val_pre_ratings, un_val_ratings, ['MSE', 'NLL', 'AUC'])

            # validation metrics on bis
            bi_val_pre_ratings = torch.empty(0).to(device)
            bi_val_ratings = torch.empty(0).to(device)
            for batch_idx, (users, items, ratings) in enumerate(biasval_loader):
                pre_ratings = CFF_model(users, items)
                bi_val_pre_ratings = torch.cat((bi_val_pre_ratings, pre_ratings))
                bi_val_ratings = torch.cat((bi_val_ratings, ratings))

            bi_val_results = utils.metrics.evaluate(bi_val_pre_ratings, bi_val_ratings, ['MSE', 'NLL', 'AUC'])
            # if math.isnan(bi_val_results['MSE']):
            #     print("#################is NAN##################")
            #     exit(0)

        print('Epoch: {0:2d} / {1}, Traning: {2}, un_Validation: {3}, bi_Validation: {4}'.format(epo, training_args['epochs'], ' '.join([key+':'+'%.3f'%train_results[key] for key in train_results]), ' '.join([key+':'+'%.3f'%un_val_results[key] for key in un_val_results]), ' '.join([key+':'+'%.3f'%bi_val_results[key] for key in bi_val_results])))
        if epo % 200 == 0:
            # test metrics on unbias
            step_test_result = step_test(test_loader, CFF_model, 'unbias', epo, args.dataset)
            # test metrics on bias
            step_test_result = step_test(biastest_loader, CFF_model, 'bias', epo, args.dataset)

        if early_stopping_cff.check([un_val_results['AUC'], bi_val_results['AUC']], epo):
            fitlog.add_best_metric({"bias_val":{"Earlystop":epo}})
            break
    
    # restore best model
    print('Loading {}th epoch'.format(early_stopping_cff.best_epoch))
    CFF_model.load_state_dict(early_stopping_cff.best_state)
    try:
        torch.save(CFF_model.state_dict(), os.path.join(f"{fitlog.get_log_folder(absolute=True)}/CFF-label_{str(time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time())))}.pth.tar"))
    except:
        print(f'########cant save model at{fitlog.get_log_folder(absolute=True)}#############')


    # validation metrics on unbias
    val_pre_ratings = torch.empty(0).to(device)
    val_ratings = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(val_loader):
        pre_ratings = CFF_model(users, items)
        val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
        val_ratings = torch.cat((val_ratings, ratings))
    val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])
    fitlog.add_best_metric({"CFF_unbias_val":{"MSE":val_results['MSE'], "NLL":val_results['NLL'], "AUC":val_results['AUC']}})

    # test metrics on unbias
    print('#'*30)
    if args.dataset == 'kuai':
        CFF_unbias_result = both_test(test_loader, CFF_model, ('CFF', 'CFF', 'unbias'), dataset=args.dataset)
    else:
        CFF_unbias_result = both_test(test_loader, CFF_model, ('CFF', 'CFF', 'unbias'))
    print('The CFF performance of unbias validation set: {}'.format(' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))
    print('#'*30)


    # validation metrics on bias
    val_pre_ratings = torch.empty(0).to(device)
    val_ratings = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(biasval_loader):
        pre_ratings = CFF_model(users, items)
        val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
        val_ratings = torch.cat((val_ratings, ratings))
    val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])
    fitlog.add_best_metric({"CFF_bias_val":{"MSE":val_results['MSE'], "NLL":val_results['NLL'], "AUC":val_results['AUC']}})

    # test metrics on bias
    print('#'*30)
    if args.dataset == 'kuai':
        CFF_bias_result = both_test(biastest_loader, CFF_model, ('CFF', 'CFF', 'bias'), dataset=args.dataset)
    else:
        CFF_bias_result = both_test(biastest_loader, CFF_model, ('CFF', 'CFF', 'bias'))
    print('The CFF performance of bias validation set: {}'.format(' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))
    print('#'*30)

if __name__ == "__main__": 
    # time.sleep(5)
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    # fitlog.debug()
    # args.exp_name = str(time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time())))
    args.exp_name = 'stable'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if  os.path.exists("/home/"+args.type+'/'+args.dataset+'/'):
        fitlog.set_log_dir("/home/"+args.type+'/'+args.dataset+'/')
    else:
        os.makedirs("/home/"+args.type+'/'+args.dataset+'/')
        fitlog.set_log_dir("/home/"+args.type+'/'+args.dataset+'/')
    fitlog.add_hyper(args)
    bias_train, bias_validation, bias_test, unif_train, unif_validation, unif_test, m, n = utils.load_dataset.load_dataset(data_name=args.dataset, type = args.type, seed = args.seed, device=device)
    train_and_eval_CFF(bias_train+unif_train, bias_validation, bias_test, unif_validation, unif_test, m, n, base_model_args = args.base_model_args, training_args = args.training_args, weightName = args.exp_name, gama = args.gama, args=args)

    fitlog.finish()
