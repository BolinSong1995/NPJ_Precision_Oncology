clear;close all
load('CCF_OPC_Outcome_clinical_stage12_final.mat');addpath(genpath('CoxScore'));S = 2;C = 3; 
%% find index of stages within cohorts
%ccf cohort
ccf_stage1 = find(strcmp(ccf_stage, 'I'));
ccf_stage2 = find(strcmp(ccf_stage, 'II'));
ccf_stage12 = sort([ccf_stage1;ccf_stage2]);
ccf_stage3 = find(strcmp(ccf_stage, 'III'));

ccf_stage12_chemoradio = intersect(aa_chemoradio_ccf,ccf_stage12);
ccf_stage3_chemoradio = intersect(aa_chemoradio_ccf,ccf_stage3);
ccf_stage12_radio = intersect(aa_radio_ccf,ccf_stage12);
ccf_stage3_radio = intersect(aa_radio_ccf,ccf_stage3);

%tcia-opc cohort
opc_stage1 = find(strcmp(tcia_opc_stage, 'I'));
opc_stage2 = find(strcmp(tcia_opc_stage, 'II'));
opc_stage12 = sort([opc_stage1;opc_stage2]);
opc_stage3 = find(strcmp(tcia_opc_stage, 'III'));

opc_stage12_chemoradio = intersect(aa_chemoradio_opc,opc_stage12);
opc_stage1_chemoradio = intersect(aa_chemoradio_opc,opc_stage1);
opc_stage2_chemoradio = intersect(aa_chemoradio_opc,opc_stage2);
opc_stage12_radio = intersect(aa_radio_opc,opc_stage12);
opc_stage1_radio = intersect(aa_radio_opc,opc_stage1);
opc_stage2_radio = intersect(aa_radio_opc,opc_stage2);

%tcia-hnscc cohort
hnscc_stage1 = find(strcmp(tcia_hnscc_stage, 'I'));
hnscc_stage2 = find(strcmp(tcia_hnscc_stage, 'II'));
hnscc_stage12 = sort([hnscc_stage1;hnscc_stage2]);
hnscc_stage3 = find(strcmp(tcia_hnscc_stage, 'III'));

hnscc_stage12_chemoradio = intersect(aa_chemoradio_hnscc,hnscc_stage12);
hnscc_stage1_chemoradio = intersect(aa_chemoradio_hnscc,hnscc_stage1);
hnscc_stage2_chemoradio = intersect(aa_chemoradio_hnscc,hnscc_stage2);
hnscc_stage12_radio = intersect(aa_radio_hnscc,hnscc_stage12);
hnscc_stage1_radio = intersect(aa_radio_hnscc,hnscc_stage1);
hnscc_stage2_radio = intersect(aa_radio_hnscc,hnscc_stage2);

%tcia-petct cohort
petct_stage1 = find(strcmp(tcia_petct_stage, 'I'));
petct_stage2 = find(strcmp(tcia_petct_stage, 'II'));
petct_stage12 = sort([petct_stage1;petct_stage2]);
petct_stage3 = find(strcmp(tcia_petct_stage, 'III'));

petct_stage12_chemoradio = intersect(aa_chemoradio_petct,petct_stage12);
petct_stage1_chemoradio = intersect(aa_chemoradio_petct,petct_stage1);
petct_stage2_chemoradio = intersect(aa_chemoradio_petct,petct_stage2);
petct_stage12_radio = intersect(aa_radio_petct,petct_stage12);
petct_stage1_radio = intersect(aa_radio_petct,petct_stage1);
petct_stage2_radio = intersect(aa_radio_petct,petct_stage2);

%% D1 - radio training
data_pos_train = feats_tcia_OPC_radiomic(opc_stage12_radio(1:60),:);
surv_pos_train = cell2mat(Outcome_tcia_OPC_pos(opc_stage12_radio(1:60),2));
censor_pos_train = 1 - cell2mat(Outcome_tcia_OPC_pos(opc_stage12_radio(1:60),3));
os_train = cell2mat(Outcome_tcia_OPC_pos(opc_stage12_radio(1:60),S));
os_censor_train = 1 - cell2mat(Outcome_tcia_OPC_pos(opc_stage12_radio(1:60),C));

[risk_threshold, beta, cur_selected_idx, num_top_feats, labels_pred, rs_train]...
    = surv_train(data_pos_train,surv_pos_train,censor_pos_train);
labels_cell = cell(length(labels_pred),1);labels_cell(labels_pred) = {'High-pRiS'};labels_cell(~labels_pred) = {'Low-pRiS'};
MatSurvHM_C(os_train, 1-os_censor_train, labels_cell, rs_train...
    , 'XLim', 90,'FlipGroupOrder',1,  'LineColor','Lancet', 'BaseFontSize', 11,'InvHR',1);

%% D2 - internal validation

opc_internal_idx = setdiff(opc_stage12,opc_stage12_radio(1:60));

surv_opc_test = cell2mat(Outcome_tcia_OPC_pos(opc_internal_idx,S));
censor_opc_test = 1 - cell2mat(Outcome_tcia_OPC_pos(opc_internal_idx,C));

[labels_pred_opc,rs_opc_val] = surv_test(bayes_combat_data_d2, ...
    beta, cur_selected_idx, surv_opc_test, censor_opc_test, risk_threshold);
labels_cell = cell(length(labels_pred_opc),1);labels_cell(labels_pred_opc) = {'High-pRiS'};labels_cell(~labels_pred_opc) = {'Low-pRiS'};
MatSurvHM_C(surv_opc_test, 1-censor_opc_test, labels_cell,rs_opc_val...
    , 'XLim', 90, 'FlipGroupOrder',1,'LineColor','Lancet', 'BaseFontSize', 11,'InvHR',1);


%% D3 - external validation
data_ext_test = cat(1,feats_OPC_CCF_120(ccf_stage12,:),feats_tcia_hnscc(hnscc_stage12,:),feats_tcia_petct(petct_stage12,:));
Outcome_tcia = cat(1,Outcome_CCF_pos_120(ccf_stage12,:),Outcome_TCIA_hnscc(hnscc_stage12,1:6),Outcome_tcia_petct_pos(petct_stage12,1:6));
surv_ext_test = cell2mat(Outcome_tcia(:,S));

censor_ext_test = 1 - cell2mat(Outcome_tcia(:,C));
[labels_pred_test,rs_val] = surv_test(data_ext_test, ...
    beta, cur_selected_idx, surv_ext_test, censor_ext_test, risk_threshold);
labels_cell = cell(length(labels_pred_test),1);labels_cell(labels_pred_test) = {'High-pRiS'};labels_cell(~labels_pred_test) = {'Low-pRiS'};
MatSurvHM_C(surv_ext_test, 1-censor_ext_test, labels_cell,rs_val...
    , 'XLim', 90, 'FlipGroupOrder',1,'LineColor','Lancet', 'BaseFontSize', 11,'InvHR',1);

%% feature map {'median-Laws S5W5'} and {'std15mmCoLlAGe energy'} on 'CCFOP140'(Low-pRiS) and 'ccfop287'(high-pRiS)
addpath(genpath('/Users/bolin/Documents/MATLAB/feature_extraction'))
addpath(genpath('/Users/bolin/Documents/MATLAB/peritumoral-radiomics-master'))
gunzip('/Users/bolin/Desktop/All_Data/p16_pos_oropharyngeal/CCF_OPC/images/cc18-HN208/CT_img.nii.gz')
V_lr = nii_read_volume(nii_read_header('/Users/bolin/Desktop/All_Data/p16_pos_oropharyngeal/CCF_OPC/images/cc18-HN208/CT_img.nii'));
V_lr = double(rot90(fliplr(V_lr)));
gunzip('/Users/bolin/Desktop/All_Data/p16_pos_oropharyngeal/CCF_OPC/images/cc18-HN208/label.nii.gz')
M_lr = nii_read_volume(nii_read_header('/Users/bolin/Desktop/All_Data/p16_pos_oropharyngeal/CCF_OPC/images/cc18-HN208/label.nii'));
M_lr = double(rot90(fliplr(M_lr)));
[featints_lr_intra,featnames_lr_intra,~,~] = extract2DFeatureInfo(V_lr,M_lr,'collage',3);
feature_map(V_lr, M_lr, featints_lr_intra{2});

%%
gunzip('/Users/bolin/Desktop/All_Data/p16_pos_oropharyngeal/CCF_OPC/images/cc18-HN175/CT_img.nii.gz')
V_hr = nii_read_volume(nii_read_header('/Users/bolin/Desktop/All_Data/p16_pos_oropharyngeal/CCF_OPC/images/cc18-HN175/CT_img.nii'));
V_hr = double(rot90(fliplr(V_hr)));
gunzip('/Users/bolin/Desktop/All_Data/p16_pos_oropharyngeal/CCF_OPC/images/cc18-HN175/label.nii.gz')
M_hr = nii_read_volume(nii_read_header('/Users/bolin/Desktop/All_Data/p16_pos_oropharyngeal/CCF_OPC/images/cc18-HN175/label.nii'));
M_hr = double(rot90(fliplr(M_hr)));
[featints_hr_intra,featnames_hr_intra,~,~] = extract2DFeatureInfo(V_hr,M_hr,'collage',3);
feature_map(V_hr, M_hr, featints_hr_intra{2})

%% peritumoral feature map
slice = 6; 
% Re-extract features, we will use the optional 3rd output to plot features 
[features, distances, matrix_indices] = ...
    intra_and_peritumoral_texture_with_distance(V_lr, M_lr, 1, [], 15);

I = V_lr(:,:,slice); % slice of interest
% Reconstruct 3D feature volume from vectors of voxelwise values and their
% corresponding indices
[feature_maps] = feature_maps_from_values_and_indices(features, ... 
    matrix_indices, size(V_lr));
ftr = 42; % collage energy
collage_map = feature_maps(:,:,slice,ftr); 
% Do the same with the distances from the tumor - will be used to generate 
% peritumoral mask
[distance_map] = feature_maps_from_values_and_indices(distances, matrix_indices, size(V_hr));
p_mask = distance_map(:,:,slice) > 0 & distance_map(:,:,slice) <= 5;
p_mask2 = distance_map(:,:,slice) > 5 & distance_map(:,:,slice) <= 10;
p_mask3 = distance_map(:,:,slice) > 10 & distance_map(:,:,slice) <= 15;
p_mask4 = distance_map(:,:,slice) > 0 & distance_map(:,:,slice) <= 15;
% scale collage map between 0 and 1 - needed for overlay function
collage_map_scaled = (collage_map - min(collage_map(p_mask))) ... 
    /(max(collage_map(p_mask))-min(collage_map(p_mask)));
collage_map_scaled2 = (collage_map - min(collage_map(p_mask2))) ... 
    /(max(collage_map(p_mask2))-min(collage_map(p_mask2)));
collage_map_scaled3 = (collage_map - min(collage_map(p_mask3))) ... 
    /(max(collage_map(p_mask3))-min(collage_map(p_mask3)));
collage_map_scaled4 = (collage_map - min(collage_map(p_mask4))) ... 
    /(max(collage_map(p_mask4))-min(collage_map(p_mask4)));
% plot peritumoral collage expression
overlayProbMap(I,p_mask,p_mask2,p_mask3,p_mask4,collage_map_scaled4,1)


%% D2+D3 combined - stage I
opc_internal_stage1_idx = setdiff(opc_internal_idx,opc_stage2);

data_combined_stage1 = cat(1,feats_tcia_OPC_radiomic(opc_internal_stage1_idx,:),feats_OPC_CCF_120(ccf_stage1,:),...
    feats_tcia_hnscc(hnscc_stage1,:),feats_tcia_petct(petct_stage1,:));
Outcome_combined_stage1 = cat(1,Outcome_tcia_OPC_pos(opc_internal_stage1_idx,1:6),Outcome_CCF_pos_120(ccf_stage1,:),...
    Outcome_TCIA_hnscc(hnscc_stage1,1:6),Outcome_tcia_petct_pos(petct_stage1,1:6));
surv_combined_stage1 = cell2mat(Outcome_combined_stage1(:,S));

censor_combined_stage1 = 1 - cell2mat(Outcome_combined_stage1(:,C));
[labels_pred_combined_stage1,rs_combined_stage1] = surv_test(data_combined_stage1, ...
    beta, cur_selected_idx, surv_combined_stage1, censor_combined_stage1, risk_threshold);
labels_cell = cell(length(labels_pred_combined_stage1),1);labels_cell(labels_pred_combined_stage1) = {'High-pRiS'};labels_cell(~labels_pred_combined_stage1) = {'Low-pRiS'};
MatSurvHM_C(surv_combined_stage1, 1-censor_combined_stage1, labels_cell,rs_combined_stage1...
    , 'XLim', 90, 'FlipGroupOrder',1,'LineColor','Lancet', 'BaseFontSize', 11,'InvHR',1);


%% D2+D3 combined - stage II
opc_internal_stage2_idx = setdiff(opc_internal_idx,opc_stage1);

data_combined_stage2 = cat(1,feats_tcia_OPC_radiomic(opc_internal_stage2_idx,:),feats_OPC_CCF_120(ccf_stage2,:),...
    feats_tcia_hnscc(hnscc_stage2,:),feats_tcia_petct(petct_stage2,:));
Outcome_combined_stage2 = cat(1,Outcome_tcia_OPC_pos(opc_internal_stage2_idx,1:6),Outcome_CCF_pos_120(ccf_stage2,:),...
    Outcome_TCIA_hnscc(hnscc_stage2,1:6),Outcome_tcia_petct_pos(petct_stage2,1:6));
surv_combined_stage2 = cell2mat(Outcome_combined_stage2(:,S));

censor_combined_stage2 = 1 - cell2mat(Outcome_combined_stage2(:,C));
[labels_pred_combined_stage2,rs_combined_stage2] = surv_test(data_combined_stage2, ...
    beta, cur_selected_idx, surv_combined_stage2, censor_combined_stage2, risk_threshold);
labels_cell = cell(length(labels_pred_combined_stage2),1);labels_cell(labels_pred_combined_stage2) = {'High-pRiS'};labels_cell(~labels_pred_combined_stage2) = {'Low-pRiS'};
MatSurvHM_C(surv_combined_stage2, 1-censor_combined_stage2, labels_cell,rs_combined_stage2...
    , 'XLim', 90, 'FlipGroupOrder',1,'LineColor','Lancet', 'BaseFontSize', 11,'InvHR',1);

%% D2 - radiotherapy alone
% thresh1 = prctile(rs_train,66);
% thresh2 = prctile(rs_train,33);

%thresh3 = prctile(rs_train,50);
thresh3 = -1.1;

data_radio_opc = feats_tcia_OPC_radiomic(opc_stage12_radio(61:end),:);
Outcome_radio_opc = Outcome_tcia_OPC_pos(opc_stage12_radio(61:end),1:6);
surv_radio_opc = cell2mat(Outcome_radio_opc(:,S));

censor_radio_opc = 1 - cell2mat(Outcome_radio_opc(:,C));
[labels_radio_opc,rs_radio_opc] = surv_test(data_radio_opc, ...
    beta, cur_selected_idx, surv_radio_opc, censor_radio_opc, risk_threshold);
labels_cell = cell(length(labels_radio_opc),1);labels_cell(:) = {'High-pRiS'};labels_cell(rs_radio_opc>thresh3) = {'High-pRiS'};
labels_cell(rs_radio_opc<thresh3) = {'Low-pRiS'};
MatSurvHM(surv_radio_opc, 1-censor_radio_opc, labels_cell...
    , 'XLim', 90, 'LineColor','Lancet','FlipGroupOrder',1, 'BaseFontSize', 11,'InvHR',1,'FlipColorOrder',0);

%% D2 - chemoradiotherapy
data_chemoradio_opc = feats_tcia_OPC_radiomic(opc_stage12_chemoradio,:);
Outcome_chemoradio_opc = Outcome_tcia_OPC_pos(opc_stage12_chemoradio,1:6);
surv_chemoradio_opc = cell2mat(Outcome_chemoradio_opc(:,S));

censor_chemoradio_opc = 1 - cell2mat(Outcome_chemoradio_opc(:,C));
[labels_chemoradio_opc,rs_chemoradio_opc] = surv_test(data_chemoradio_opc, ...
    beta, cur_selected_idx, surv_chemoradio_opc, censor_chemoradio_opc, risk_threshold);
labels_cell = cell(length(labels_chemoradio_opc),1);labels_cell(:) = {'High-pRiS'};labels_cell(rs_chemoradio_opc>thresh3) = {'High-pRiS'};
labels_cell(rs_chemoradio_opc<thresh3) = {'Low-pRiS'};
MatSurvHM(surv_chemoradio_opc, 1-censor_chemoradio_opc, labels_cell...
    , 'XLim', 90, 'LineColor','Lancet','FlipGroupOrder',1, 'BaseFontSize', 11,'InvHR',1,'FlipColorOrder',0);

%% D2 - high risk group
aa_hr_opc = find(rs_opc_val>thresh3);
aa_lr_opc = find(rs_opc_val<thresh3);

trx_opc_stage12 = aa_trx_opc(opc_internal_idx);
trx_opc_stage12_hr = trx_opc_stage12(aa_hr_opc);

Outcome_OPC_stage12 = Outcome_tcia_OPC_pos(opc_internal_idx,:);
surv_opc_hr = cell2mat(Outcome_OPC_stage12(aa_hr_opc,S));
censor_opc_hr = 1 - cell2mat(Outcome_OPC_stage12(aa_hr_opc,C));

MatSurvHM(surv_opc_hr, 1-censor_opc_hr, trx_opc_stage12_hr...
    , 'XLim', 60, 'FlipGroupOrder',1,'LineColor','JCO', 'BaseFontSize', 11,'InvHR',0,...
    'KM_position',[0.28 0.3 0.6 0.6],'RT_position',[0.28 0.05 0.6 0.12]);

%% D2 - low risk group
trx_opc_stage12_lr = trx_opc_stage12(aa_lr_opc);

Outcome_OPC_stage12 = Outcome_tcia_OPC_pos(opc_internal_idx,:);
surv_opc_lr = cell2mat(Outcome_OPC_stage12(aa_lr_opc,S));
censor_opc_lr = 1 - cell2mat(Outcome_OPC_stage12(aa_lr_opc,C));

MatSurvHM(surv_opc_lr, 1-censor_opc_lr, trx_opc_stage12_lr...
    , 'XLim', 60, 'FlipGroupOrder',1,'LineColor','JCO', 'BaseFontSize', 11,'InvHR',0,...
    'KM_position',[0.28 0.3 0.6 0.6],'RT_position',[0.28 0.05 0.6 0.12]);

%% D3 - radiotherapy alone
thresh3 = -1.1;
data_radio_ccf = feats_OPC_CCF_120(ccf_stage12_radio,:);
Outcome_radio_ccf = Outcome_CCF_pos_120(ccf_stage12_radio,1:6);
surv_radio_ccf = cell2mat(Outcome_radio_ccf(:,S));
censor_radio_ccf = 1 - cell2mat(Outcome_radio_ccf(:,C));

data_radio_hnscc = feats_tcia_hnscc(hnscc_stage12_radio,:);
Outcome_radio_hnscc = Outcome_TCIA_hnscc(hnscc_stage12_radio,1:6);
surv_radio_hnscc = cell2mat(Outcome_radio_hnscc(:,S));
censor_radio_hnscc = 1 - cell2mat(Outcome_radio_hnscc(:,C));

data_radio_petct = feats_tcia_petct(petct_stage12_radio,:);
Outcome_radio_petct = Outcome_tcia_petct_pos(petct_stage12_radio,1:6);
surv_radio_petct = cell2mat(Outcome_radio_petct(:,S));
censor_radio_petct = 1 - cell2mat(Outcome_radio_petct(:,C));

data_radio_val = cat(1,data_radio_ccf,data_radio_hnscc,data_radio_petct);
surv_radio_val = cat(1,surv_radio_ccf,surv_radio_hnscc,surv_radio_petct);
censor_radio_val = cat(1,censor_radio_ccf,censor_radio_hnscc,censor_radio_petct);

[labels_radio_val,rs_radio_val] = surv_test(data_radio_val, ...
    beta, cur_selected_idx, surv_radio_val, censor_radio_val, risk_threshold);
labels_cell = cell(length(labels_radio_val),1);labels_cell(:) = {'High-pRiS'};labels_cell(rs_radio_val>thresh3) = {'High-pRiS'};
labels_cell(rs_radio_val<thresh3) = {'Low-pRiS'};
MatSurvHM(surv_radio_val, 1-censor_radio_val, labels_cell...
    , 'XLim', 90, 'LineColor','Lancet','FlipGroupOrder',1, 'BaseFontSize', 11,'InvHR',1,'FlipColorOrder',0);


%% D3 - chemoradio
data_chemoradio_ccf = feats_OPC_CCF_120(ccf_stage12_chemoradio,:);
Outcome_chemoradio_ccf = Outcome_CCF_pos_120(ccf_stage12_chemoradio,1:6);
surv_chemoradio_ccf = cell2mat(Outcome_chemoradio_ccf(:,S));
censor_chemoradio_ccf = 1 - cell2mat(Outcome_chemoradio_ccf(:,C));

data_chemoradio_hnscc = feats_tcia_hnscc(hnscc_stage12_chemoradio,:);
Outcome_chemoradio_hnscc = Outcome_TCIA_hnscc(hnscc_stage12_chemoradio,1:6);
surv_chemoradio_hnscc = cell2mat(Outcome_chemoradio_hnscc(:,S));
censor_chemoradio_hnscc = 1 - cell2mat(Outcome_chemoradio_hnscc(:,C));

data_chemoradio_petct = feats_tcia_petct(petct_stage12_chemoradio,:);
Outcome_chemoradio_petct = Outcome_tcia_petct_pos(petct_stage12_chemoradio,1:6);
surv_chemoradio_petct = cell2mat(Outcome_chemoradio_petct(:,S));
censor_chemoradio_petct = 1 - cell2mat(Outcome_chemoradio_petct(:,C));

data_chemoradio_val = cat(1,data_chemoradio_ccf,data_chemoradio_hnscc,data_chemoradio_petct);
surv_chemoradio_val = cat(1,surv_chemoradio_ccf,surv_chemoradio_hnscc,surv_chemoradio_petct);
censor_chemoradio_val = cat(1,censor_chemoradio_ccf,censor_chemoradio_hnscc,censor_chemoradio_petct);

[labels_chemoradio_val,rs_chemoradio_val] = surv_test(data_chemoradio_val, ...
    beta, cur_selected_idx, surv_chemoradio_val, censor_chemoradio_val, risk_threshold);
labels_cell = cell(length(labels_chemoradio_val),1);labels_cell(:) = {'High-pRiS'};labels_cell(rs_chemoradio_val>thresh3) = {'High-pRiS'};
labels_cell(rs_chemoradio_val<thresh3) = {'Low-pRiS'};
MatSurvHM(surv_chemoradio_val, 1-censor_chemoradio_val, labels_cell...
    , 'XLim', 90, 'LineColor','Lancet','FlipGroupOrder',1, 'BaseFontSize', 11,'InvHR',1,'FlipColorOrder',0);


%% D3 - high risk group
aa_hr_val = find(rs_val>thresh3);
aa_lr_val = find(rs_val<thresh3);

trx_ccf_stage12 = aa_trx_ccf(ccf_stage12);
trx_hnscc_stage12 = aa_trx_hnscc(hnscc_stage12);
trx_petct_stage12 = aa_trx_petct(petct_stage12);
trx_val_stage12 = cat(1,trx_ccf_stage12,trx_hnscc_stage12,trx_petct_stage12);

trx_val_stage12_hr = trx_val_stage12(aa_hr_val);

Outcome_ccf_stage12 = Outcome_CCF_pos_120(ccf_stage12,:);
Outcome_hnscc_stage12 = Outcome_TCIA_hnscc(hnscc_stage12,1:6);
Outcome_petct_stage12 = Outcome_tcia_petct_pos(petct_stage12,1:6);
Outcome_val_stage12 = cat(1,Outcome_ccf_stage12,Outcome_hnscc_stage12,Outcome_petct_stage12);

surv_val_hr = cell2mat(Outcome_val_stage12(aa_hr_val,S));
censor_val_hr = 1 - cell2mat(Outcome_val_stage12(aa_hr_val,C));

MatSurvHM(surv_val_hr, 1-censor_val_hr, trx_val_stage12_hr...
    , 'XLim', 60, 'FlipGroupOrder',1,'LineColor','JCO', 'BaseFontSize', 11,'InvHR',0,...
    'KM_position',[0.28 0.3 0.6 0.6],'RT_position',[0.28 0.05 0.6 0.12]);

%% D3 - low risk group
trx_val_stage12_lr = trx_val_stage12(aa_lr_val);

surv_val_lr = cell2mat(Outcome_val_stage12(aa_lr_val,S));
censor_val_lr = 1 - cell2mat(Outcome_val_stage12(aa_lr_val,C));

MatSurvHM(surv_val_lr, 1-censor_val_lr, trx_val_stage12_lr...
    , 'XLim', 60, 'FlipGroupOrder',1,'LineColor','JCO', 'BaseFontSize', 11,'InvHR',0,...
    'KM_position',[0.28 0.3 0.6 0.6],'RT_position',[0.28 0.05 0.6 0.12]);


%% output data for treatment interaction test
%% D1 patients
radio_train = opc_stage12_radio(1:60);
ID = Outcome_tcia_OPC_pos(radio_train,1);
OS = [Outcome_tcia_OPC_pos(radio_train,2)];OS_status = [Outcome_tcia_OPC_pos(radio_train,3)];
DFS = [Outcome_tcia_OPC_pos(radio_train,4)];DFS_status = [Outcome_tcia_OPC_pos(radio_train,5)];
trt = zeros(60,1);

predictive_labels = zeros(48,1);
predictive_labels(rs_train>thresh3) = 1;
pRiS = rs_train;
D1_interaction = table(ID,OS,OS_status,DFS,DFS_status,trt,predictive_labels,pRiS);

%% D2 patients
ID = [Outcome_radio_opc(:,1);Outcome_chemoradio_opc(:,1)];
OS = [Outcome_radio_opc(:,2);Outcome_chemoradio_opc(:,2)];OS_status = [Outcome_radio_opc(:,3);Outcome_chemoradio_opc(:,3)];
DFS = [Outcome_radio_opc(:,4);Outcome_chemoradio_opc(:,4)];DFS_status = [Outcome_radio_opc(:,5);Outcome_chemoradio_opc(:,5)];
trt = zeros(162,1);trt(49:end) = 1;

predictive_radio = zeros(48,1);
predictive_radio(rs_radio_opc>thresh3) = 1;
predictive_chemoradio = zeros(114,1);
predictive_chemoradio(rs_chemoradio_opc>thresh3) = 1;
predictive_labels = [predictive_radio;predictive_chemoradio];
pRiS = [rs_radio_opc;rs_chemoradio_opc];
D2_interaction = table(ID,OS,OS_status,DFS,DFS_status,trt,predictive_labels,pRiS);


%% D3 patients
ID = [Outcome_radio_ccf(:,1);Outcome_radio_hnscc(:,1);Outcome_radio_petct(:,1);Outcome_chemoradio_ccf(:,1);...
    Outcome_chemoradio_hnscc(:,1);Outcome_chemoradio_petct(:,1)];
OS = [Outcome_radio_ccf(:,2);Outcome_radio_hnscc(:,2);Outcome_radio_petct(:,2);Outcome_chemoradio_ccf(:,2);...
    Outcome_chemoradio_hnscc(:,2);Outcome_chemoradio_petct(:,2)];
OS_status = [Outcome_radio_ccf(:,3);Outcome_radio_hnscc(:,3);Outcome_radio_petct(:,3);Outcome_chemoradio_ccf(:,3);...
    Outcome_chemoradio_hnscc(:,3);Outcome_chemoradio_petct(:,3)];
DFS = [Outcome_radio_ccf(:,4);Outcome_radio_hnscc(:,4);Outcome_radio_petct(:,4);Outcome_chemoradio_ccf(:,4);...
    Outcome_chemoradio_hnscc(:,4);Outcome_chemoradio_petct(:,4)];
DFS_status = [Outcome_radio_ccf(:,5);Outcome_radio_hnscc(:,5);Outcome_radio_petct(:,5);Outcome_chemoradio_ccf(:,5);...
    Outcome_chemoradio_hnscc(:,5);Outcome_chemoradio_petct(:,5)];
trt = zeros(269,1);trt(55:end) = 1;

predictive_radio = zeros(54,1);
predictive_radio(rs_radio_val>thresh3) = 1;
predictive_chemoradio = zeros(215,1);
predictive_chemoradio(rs_chemoradio_val>thresh3) = 1;
predictive_labels = [predictive_radio;predictive_chemoradio];
pRiS = [rs_radio_val;rs_chemoradio_val];
D3_interaction = table(ID,OS,OS_status,DFS,DFS_status,trt,predictive_labels,pRiS);

%% D1 clinicopathologic variables
idx = [];
for i = 1:60
    idx(i) = find(strcmp(a_clinical_tcia_OPC_pos(:,1),string(D1_interaction.ID{i})));
end
idx = idx';

D1_interaction.age = cell2mat(a_clinical_tcia_OPC_pos(idx,2));
D1_interaction.PY = cell2mat(a_clinical_tcia_OPC_pos(idx,5));
D1_interaction.gender = a_clinical_tcia_OPC_pos(idx,3);
D1_interaction.subsite = a_clinical_tcia_OPC_pos(idx,9);
D1_interaction.smoking = a_clinical_tcia_OPC_pos(idx,6);
D1_interaction.Nstage = a_clinical_tcia_OPC_pos(idx,35);
D1_interaction.Tstage = a_clinical_tcia_OPC_pos(idx,10);
D1_interaction.overallstage = a_clinical_tcia_OPC_pos(idx,36);

D1_interaction.subsite(strcmp(D1_interaction.subsite,'Soft Palate')) = {'Posterior Wall'};
D1_interaction.subsite(strcmp(D1_interaction.subsite,'lat wall')) = {'Posterior Wall'};
D1_interaction.subsite(strcmp(D1_interaction.subsite,'post wall')) = {'Posterior Wall'};
D1_interaction.subsite(strcmp(D1_interaction.subsite,'Tonsil Pillar')) = {'Tonsil'};
D1_interaction.subsite(strcmp(D1_interaction.subsite,'Tonsillar Fossa')) = {'Tonsil'};
D1_interaction.subsite(strcmp(D1_interaction.subsite,'Vallecula')) = {'Base of Tongue'};
D1_interaction.smoking(strcmp(D1_interaction.smoking,'Ex-smoker')) = {'Former'};
D1_interaction.smoking(strcmp(D1_interaction.smoking,'Non-smoker')) = {'Never'};

%% D2 clinicopathologic variables
idx = [];
for i = 1:162
    idx(i) = find(strcmp(a_clinical_tcia_OPC_pos(:,1),string(D2_interaction.ID{i})));
end
idx = idx';

D2_interaction.age = cell2mat(a_clinical_tcia_OPC_pos(idx,2));
D2_interaction.PY = cell2mat(a_clinical_tcia_OPC_pos(idx,5));
D2_interaction.gender = a_clinical_tcia_OPC_pos(idx,3);
D2_interaction.subsite = a_clinical_tcia_OPC_pos(idx,9);
D2_interaction.smoking = a_clinical_tcia_OPC_pos(idx,6);
D2_interaction.Nstage = a_clinical_tcia_OPC_pos(idx,35);
D2_interaction.Tstage = a_clinical_tcia_OPC_pos(idx,10);
D2_interaction.overallstage = a_clinical_tcia_OPC_pos(idx,36);

D2_interaction.subsite(strcmp(D2_interaction.subsite,'Soft Palate')) = {'Posterior Wall'};
D2_interaction.subsite(strcmp(D2_interaction.subsite,'lat wall')) = {'Posterior Wall'};
D2_interaction.subsite(strcmp(D2_interaction.subsite,'post wall')) = {'Posterior Wall'};
D2_interaction.subsite(strcmp(D2_interaction.subsite,'Tonsil Pillar')) = {'Tonsil'};
D2_interaction.subsite(strcmp(D2_interaction.subsite,'Tonsillar Fossa')) = {'Tonsil'};
D2_interaction.subsite(strcmp(D2_interaction.subsite,'Vallecula')) = {'Base of Tongue'};
D2_interaction.smoking(strcmp(D2_interaction.smoking,'Ex-smoker')) = {'Former'};
D2_interaction.smoking(strcmp(D2_interaction.smoking,'Non-smoker')) = {'Never'};

%% D3 clinicopathologic variables
idx = [];
for i = 1:11
    idx(i) = find(strcmp(a_CCF_clinical_info_120.DEID,string(D3_interaction.ID{i})))
end
idx = idx';

D3_interaction.age(1:11) = a_CCF_clinical_info_120.age(idx);
D3_interaction.PY(1:11) = a_CCF_clinical_info_120.pack_years(idx);
D3_interaction.PY(isnan(D3_interaction.PY)) = 0;

D3_interaction.gender(1:11) = cellstr(string(a_CCF_clinical_info_120.sex(idx)));
D3_interaction.subsite(1:11) = cellstr(string(a_CCF_clinical_info_120.oropharynx_subsite(idx)));
D3_interaction.smoking(1:11) = cellstr(string(a_CCF_clinical_info_120.smoking_history(idx)));
D3_interaction.Nstage(1:11) = cellstr(string(a_CCF_clinical_info_120.AJCC8N(idx)));
D3_interaction.Tstage(1:11) = cellstr(string(a_CCF_clinical_info_120.AJCC8T(idx)));
D3_interaction.overallstage(1:11) = cellstr(string(a_CCF_clinical_info_120.AJCC8Staging(idx)));


idx = [];
for i = 1:37
    idx(i) = find(strcmp(a_clinical_tcia_hnscc.TCIARadiomicsID,string(D3_interaction.ID{i+11})))
end
idx = idx';

D3_interaction.age(12:48) = a_clinical_tcia_hnscc.AgeAtDiag(idx);
D3_interaction.PY(12:48) = a_clinical_tcia_hnscc.SmokingStatus_Packs_Years_(idx);

D3_interaction.gender(12:48) = cellstr(string(a_clinical_tcia_hnscc.Gender(idx)));
D3_interaction.subsite(12:48) = cellstr(string(a_clinical_tcia_hnscc.CancerSubsiteOfOrigin(idx)));
D3_interaction.smoking(12:48) = cellstr(string(a_clinical_tcia_hnscc.SmokingStatus(idx)));
D3_interaction.Nstage(12:48) = cellstr(string(a_clinical_tcia_hnscc.N_category(idx)));
D3_interaction.Tstage(12:48) = cellstr(string(a_clinical_tcia_hnscc.T_category(idx)));
D3_interaction.overallstage(12:48) = tcia_hnscc_stage(idx);


idx = [];
for i = 1:80
    idx(i) = find(strcmp(a_CCF_clinical_info_120.DEID,string(D3_interaction.ID{i+54})))
end
idx = idx';

D3_interaction.age(55:134) = a_CCF_clinical_info_120.age(idx);
D3_interaction.PY(55:134) = a_CCF_clinical_info_120.pack_years(idx);
D3_interaction.PY(isnan(D3_interaction.PY)) = 0;

D3_interaction.gender(55:134) = cellstr(string(a_CCF_clinical_info_120.sex(idx)));
D3_interaction.subsite(55:134) = cellstr(string(a_CCF_clinical_info_120.oropharynx_subsite(idx)));
D3_interaction.smoking(55:134) = cellstr(string(a_CCF_clinical_info_120.smoking_history(idx)));
D3_interaction.Nstage(55:134) = cellstr(string(a_CCF_clinical_info_120.AJCC8N(idx)));
D3_interaction.Tstage(55:134) = cellstr(string(a_CCF_clinical_info_120.AJCC8T(idx)));
D3_interaction.overallstage(55:134) = cellstr(string(a_CCF_clinical_info_120.AJCC8Staging(idx)));


idx = [];
for i = 1:97
    idx(i) = find(strcmp(a_clinical_tcia_hnscc.TCIARadiomicsID,string(D3_interaction.ID{i+134})))
end
idx = idx';

D3_interaction.age(135:231) = a_clinical_tcia_hnscc.AgeAtDiag(idx);
D3_interaction.PY(135:231) = a_clinical_tcia_hnscc.SmokingStatus_Packs_Years_(idx);

D3_interaction.gender(135:231) = cellstr(string(a_clinical_tcia_hnscc.Gender(idx)));
D3_interaction.subsite(135:231) = cellstr(string(a_clinical_tcia_hnscc.CancerSubsiteOfOrigin(idx)));
D3_interaction.smoking(135:231) = cellstr(string(a_clinical_tcia_hnscc.SmokingStatus(idx)));
D3_interaction.Nstage(135:231) = cellstr(string(a_clinical_tcia_hnscc.N_category(idx)));
D3_interaction.Tstage(135:231) = cellstr(string(a_clinical_tcia_hnscc.T_category(idx)));
D3_interaction.overallstage(135:231) = tcia_hnscc_stage(idx);

D3_interaction([49:54 232:269],:) = [] ;

D3_interaction.subsite(strcmp(D3_interaction.subsite,'Valleculae')) = {'Base of Tongue'};
D3_interaction.subsite(strcmp(D3_interaction.subsite,'NOS')) = {'Base of Tongue'};
D3_interaction.subsite(strcmp(D3_interaction.subsite,'Oropharynx, NOS')) = {'Base of Tongue'};
D3_interaction.subsite(strcmp(D3_interaction.subsite,'Base of tongue')) = {'Base of Tongue'};
D3_interaction.subsite(strcmp(D3_interaction.subsite,'Soft palate')) = {'Posterior Wall'};
D3_interaction.subsite(strcmp(D3_interaction.subsite,'Glossopharyngeal sulcus')) = {'Posterior Wall'};
D3_interaction.smoking(strcmp(D3_interaction.smoking,'Former use (Quit >3 months ago)')) = {'Former'};
D3_interaction.smoking(strcmp(D3_interaction.smoking,'Use during or after treatment')) = {'Current'};
D3_interaction.smoking(strcmp(D3_interaction.smoking,'Current use')) = {'Current'};
D3_interaction.smoking(strcmp(D3_interaction.smoking,'Never smoker')) = {'Never'};

D3_interaction.Tstage(strcmp(D3_interaction.Tstage,'1')) = {'T1'};
D3_interaction.Tstage(strcmp(D3_interaction.Tstage,'2')) = {'T2'};
D3_interaction.Tstage(strcmp(D3_interaction.Tstage,'3')) = {'T3'};
D3_interaction.Nstage(strcmp(D3_interaction.Nstage,'0')) = {'N0'};
D3_interaction.Nstage(strcmp(D3_interaction.Nstage,'1')) = {'N1'};
D3_interaction.Nstage(strcmp(D3_interaction.Nstage,'2')) = {'N2'};

writetable(D1_interaction,'/Users/bolin/Desktop/nomogram/D1_interaction.csv')
writetable(D2_interaction,'/Users/bolin/Desktop/nomogram/D2_interaction.csv')
writetable(D3_interaction,'/Users/bolin/Desktop/nomogram/D3_interaction.csv')
%% D2+D3
D23_interaction = [D2_interaction;D3_interaction];
writetable(D23_interaction,'/Users/bolin/Desktop/nomogram/D23_interaction.csv')


%% D2+D3 - stage I - high risk group
aa_hr_combined_stage1 = find(rs_combined_stage1>thresh3);
aa_lr_combined_stage1 = find(rs_combined_stage1<thresh3);


trx_opc_stage1 = aa_trx_opc(opc_internal_stage1_idx);
trx_ccf_stage1 = aa_trx_ccf(ccf_stage1);
trx_hnscc_stage1 = aa_trx_hnscc(hnscc_stage1);
trx_petct_stage1 = aa_trx_petct(petct_stage1);
trx_combined_stage1 = cat(1,trx_opc_stage1,trx_ccf_stage1,trx_hnscc_stage1,trx_petct_stage1);

trx_combined_stage1_hr = trx_combined_stage1(aa_hr_combined_stage1);

Outcome_OPC_stage1 = Outcome_tcia_OPC_pos(opc_internal_stage1_idx,:);
Outcome_ccf_stage1 = Outcome_CCF_pos_120(ccf_stage1,:);
Outcome_hnscc_stage1 = Outcome_TCIA_hnscc(hnscc_stage1,1:6);
Outcome_petct_stage1 = Outcome_tcia_petct_pos(petct_stage1,1:6);
Outcome_combined_stage1 = cat(1,Outcome_OPC_stage1(:,1:6),Outcome_ccf_stage1,Outcome_hnscc_stage1,Outcome_petct_stage1);

surv_combined_stage1_hr = cell2mat(Outcome_combined_stage1(aa_hr_combined_stage1,S));
censor_combined_stage1_hr = 1 - cell2mat(Outcome_combined_stage1(aa_hr_combined_stage1,C));

MatSurvHM(surv_combined_stage1_hr, 1-censor_combined_stage1_hr, trx_combined_stage1_hr...
    , 'XLim', 60, 'FlipGroupOrder',1,'LineColor','JCO', 'BaseFontSize', 11,'InvHR',0,...
    'KM_position',[0.28 0.3 0.6 0.6],'RT_position',[0.28 0.05 0.6 0.12]);

%% D2+D3 - stage I - low risk group
trx_combined_stage1_lr = trx_combined_stage1(aa_lr_combined_stage1);

surv_combined_stage1_lr = cell2mat(Outcome_combined_stage1(aa_lr_combined_stage1,S));
censor_combined_stage1_lr = 1 - cell2mat(Outcome_combined_stage1(aa_lr_combined_stage1,C));

MatSurvHM(surv_combined_stage1_lr, 1-censor_combined_stage1_lr, trx_combined_stage1_lr...
    , 'XLim', 60, 'FlipGroupOrder',1,'LineColor','JCO', 'BaseFontSize', 11,'InvHR',0,...
    'KM_position',[0.28 0.3 0.6 0.6],'RT_position',[0.28 0.05 0.6 0.12]);

%% D2+D3 - stage I - interaction test
D23_stage1_idx = find(double(strcmp(D23_interaction.overallstage,'I')));
D23_stage1_interaction = D23_interaction(D23_stage1_idx,:);
writetable(D23_stage1_interaction,'/Users/bolin/Desktop/nomogram/D23_stage1_interaction.csv')

%% D2+D3 - stage II - high risk group
aa_hr_combined_stage2 = find(rs_combined_stage2>thresh3);
aa_lr_combined_stage2 = find(rs_combined_stage2<thresh3);


trx_opc_stage2 = aa_trx_opc(opc_internal_stage2_idx);
trx_ccf_stage2 = aa_trx_ccf(ccf_stage2);
trx_hnscc_stage2 = aa_trx_hnscc(hnscc_stage2);
trx_petct_stage2 = aa_trx_petct(petct_stage2);
trx_combined_stage2 = cat(1,trx_opc_stage2,trx_ccf_stage2,trx_hnscc_stage2,trx_petct_stage2);

trx_combined_stage2_hr = trx_combined_stage2(aa_hr_combined_stage2);

Outcome_OPC_stage2 = Outcome_tcia_OPC_pos(opc_internal_stage2_idx,:);
Outcome_ccf_stage2 = Outcome_CCF_pos_120(ccf_stage2,:);
Outcome_hnscc_stage2 = Outcome_TCIA_hnscc(hnscc_stage2,1:6);
Outcome_petct_stage2 = Outcome_tcia_petct_pos(petct_stage2,1:6);
Outcome_combined_stage2 = cat(1,Outcome_OPC_stage2(:,1:6),Outcome_ccf_stage2,Outcome_hnscc_stage2,Outcome_petct_stage2);

surv_combined_stage2_hr = cell2mat(Outcome_combined_stage2(aa_hr_combined_stage2,S));
censor_combined_stage2_hr = 1 - cell2mat(Outcome_combined_stage2(aa_hr_combined_stage2,C));

MatSurvHM(surv_combined_stage2_hr, 1-censor_combined_stage2_hr, trx_combined_stage2_hr...
    , 'XLim', 60, 'FlipGroupOrder',1,'LineColor','JCO', 'BaseFontSize', 11,'InvHR',0,...
    'KM_position',[0.28 0.3 0.6 0.6],'RT_position',[0.28 0.05 0.6 0.12]);

%% D2+D3 - stage II - low risk group
trx_combined_stage2_lr = trx_combined_stage2(aa_lr_combined_stage2);

surv_combined_stage2_lr = cell2mat(Outcome_combined_stage2(aa_lr_combined_stage2,S));
censor_combined_stage2_lr = 1 - cell2mat(Outcome_combined_stage2(aa_lr_combined_stage2,C));

MatSurvHM(surv_combined_stage2_lr, 1-censor_combined_stage2_lr, trx_combined_stage2_lr...
    , 'XLim', 60, 'FlipGroupOrder',1,'LineColor','JCO', 'BaseFontSize', 11,'InvHR',0,...
    'KM_position',[0.28 0.3 0.6 0.6],'RT_position',[0.28 0.05 0.6 0.12]);

%% D2+D3 - stage II - interaction test
D23_stage2_idx = find(double(strcmp(D23_interaction.overallstage,'II')));
D23_stage2_interaction = D23_interaction(D23_stage2_idx,:);
writetable(D23_stage2_interaction,'/Users/bolin/Desktop/nomogram/D23_stage2_interaction.csv')