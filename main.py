import os
import yaml
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split

from layers.CNN_BiLSTM_Attention import CNN_BiLSTM_Attention
from dataset import MyDataset
from utils.plot_results import plot_cm, plot_curve
from utils.schedule import WarmupConstantSchedule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


def split_audio(inputs):
    split_inputs = []

    split_pvot = (inputs.shape[3] // (100 // 2)) - 1
    for i in range(split_pvot):
        seg_input = inputs[
            :, :, :, ((100 // 2) * i): (100 + ((100 // 2) * i))
        ]
        split_inputs.append(seg_input)
    return split_inputs


def train(net, train_loader, optimizer, criterion):
    net.train()
    running_loss, correct_cnt, total = 0.0, 0.0, 0.0

    for (inputs, labels) in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        target = torch.max(labels, 1)[1]

        split_inputs = split_audio(inputs)
        outputs, _ = net(split_inputs)

        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        del loss

        _, predict = torch.max(outputs, 1)

        correct_cnt += (predict == target).sum().item()
        total += torch.max(labels, 1)[1].size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct_cnt / total

    return net, train_loss, train_acc


def valid(net, valid_loader, criterion):
    net.eval()
    running_loss, correct_cnt, total = 0.0, 0.0, 0.0

    target_lists = torch.zeros(0, dtype=torch.long, device='cpu')
    predict_lists = torch.zeros(0, dtype=torch.long, device='cpu')

    with torch.no_grad():
        for (inputs, labels) in tqdm(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            target = torch.max(labels, 1)[1]

            split_inputs = split_audio(inputs)
            outputs, _ = net(split_inputs)

            loss = criterion(outputs, target)
            running_loss += loss.item()
            del loss

            _, predict = torch.max(outputs, 1)

            target_lists = torch.cat([target_lists, target.cpu()])
            predict_lists = torch.cat([predict_lists, predict.cpu()])

            correct_cnt += (predict == target).sum().item()
            total += torch.max(labels, 1)[1].size(0)

    val_loss = running_loss / len(valid_loader)
    val_acc = correct_cnt / total

    return net, val_loss, val_acc, predict_lists.numpy(), target_lists.numpy()


if __name__ == '__main__':
    # ================== [1] Set up ==================
    # load param
    with open('hyper_param.yaml', 'r') as file:
        config = yaml.safe_load(file.read())

    in_dir = config['dataset_setting']['in_dir']
    in_dim = config['dataset_setting']['in_dim']
    max_len = config['dataset_setting']['max_len']

    epochs = config['training_setting']['epoch']
    batch_size = config['training_setting']['batch_size']
    d_model = config['training_setting']['d_model']
    learning_rate = config['training_setting']['learning_rate']
    warmup_rate = config['training_setting']['warmup_rate']
    early_stopping = config['training_setting']['early_stopping']

    # make dir for results
    time_now = datetime.datetime.now()
    os.makedirs(
        "./results/{}/accuracy_curve".format(str(time_now.date())), exist_ok=True)
    os.makedirs(
        "./results/{}/confusion_matrix".format(str(time_now.date())), exist_ok=True)
    os.makedirs(
        "./results/{}/learning_curve".format(str(time_now.date())), exist_ok=True)
    os.makedirs(
        "./results/{}/model_param".format(str(time_now.date())), exist_ok=True)

    # training setup
    fold = KFold(n_splits=5, shuffle=False)
    cross_validation = 0
    cv_lists = []
    train_dataset = MyDataset(
        "{}/JTES_mcep/preprocess/**/**/npy/*.npy".format(in_dir),
        in_dim,
        max_len
    )

    # ================== [2] Training and Validation ==================
    print("Start training!")
    for fold_idx, (train_idx, valid_idx) in tqdm(enumerate(fold.split(train_dataset))):
        print('fold:{}\n'.format(fold_idx))

        # load NN model
        net = CNN_BiLSTM_Attention(
            d_model=d_model
        )
        net.to(device)
        if fold_idx == 0:
            print(net)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=learning_rate
        )
        scheduler = WarmupConstantSchedule(
            optimizer,
            warmup_epochs=epochs * warmup_rate
        )

        train_acc_curve = []
        valid_acc_curve = []
        train_loss_curve = []
        valid_loss_curve = []

        train_loader = data.DataLoader(
            data.Subset(train_dataset, train_idx), shuffle=True, batch_size=batch_size)
        valid_loader = data.DataLoader(
            data.Subset(train_dataset, valid_idx), shuffle=True, batch_size=batch_size)
        print("train_loader:{}".format(len(train_loader)))
        print("valid_loader:{}".format(len(valid_loader)))

        patience = 0
        for epoch in tqdm(range(epochs)):
            print("train phase")
            net, train_loss, train_acc = train(
                net, train_loader, optimizer, criterion)

            print("test phase")
            net, valid_loss, valid_acc, predict_list, target_list = valid(
                net, valid_loader, criterion)

            print('train_loss {:.8f} valid loss {:.8f} train_acc {:.8f} valid_acc {:.8f}'.format(
                train_loss, valid_loss, train_acc, valid_acc))

            scheduler.step()

            train_acc_curve.append(train_acc)
            valid_acc_curve.append(valid_acc)
            train_loss_curve.append(train_loss)
            valid_loss_curve.append(valid_loss)

            # ===== early-stopling =====
            if train_loss < valid_loss:
                patience += 1
                if patience > early_stopping:
                    break
            else:
                patience = 0
            # ==========================

        torch.save(
            net.state_dict(),
            './results/{}/model_param/SER_fold{}_{}_Param.pth'.format(
                str(time_now.date()),
                fold_idx + 1,
                time_now)
        )
        cv_lists.append(valid_acc)
        cross_validation += valid_acc

        plot_curve(
            train_acc_curve,
            valid_acc_curve,
            x_label=config['plot_acc_curve_setting']['acc_curve_x_label'],
            y_label=config['plot_acc_curve_setting']['acc_curve_y_label'],
            title=config['plot_acc_curve_setting']['acc_curve_title'],
            fold_idx=fold_idx,
            dir_path_name=str(time_now.date())
        )
        plot_curve(
            train_loss_curve,
            valid_loss_curve,
            x_label=config['plot_loss_curve_setting']['loss_curve_x_label'],
            y_label=config['plot_loss_curve_setting']['loss_curve_y_label'],
            title=config['plot_loss_curve_setting']['loss_curve_title'],
            fold_idx=fold_idx,
            dir_path_name=str(time_now.date())
        )
        plot_cm(
            target_list,
            predict_list,
            x_label=config['plot_cm_setting']['cm_x_label'],
            y_label=config['plot_cm_setting']['cm_y_label'],
            dir_path_name=str(time_now.date())
        )

    # ================== [3] Plot results of CV ==================
    print("cross validation:{}".format(cv_lists))
    print("cross validation [ave]:{}".format(cross_validation / fold.n_splits))
    print("Finished!")
