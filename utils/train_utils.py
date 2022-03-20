from tqdm import tqdm
import numpy as np
import torch
import pickle


def add_args(parser):
    parser.add_argument("--training_mode",
                        default='train_with_multi_source_pro',
                        type=str)
    parser.add_argument("--load_model",
                        default='NOT_LOAD',
                        type=str)
    parser.add_argument("--pretrain_model",
                        default='xlm-roberta-base',
                        type=str)
    parser.add_argument("--data_set",
                        default='multi_source_data_ultra',
                        type=str)
    parser.add_argument("--data_feature",
                        default='en_zh_fr_ex_gloss_img_data_main',
                        type=str)
    parser.add_argument("--batch_size",
                        default=8,
                        type=int)
    parser.add_argument("--hidden_size",
                        default=768,
                        type=int)
    parser.add_argument("--img_hidden_size",
                        default=128,
                        type=int)
    parser.add_argument("--classifier_learning_rate",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate for classifier.")
    parser.add_argument("--img_classifier_learning_rate",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate for img classifier.")
    parser.add_argument("--encoder_learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for encoder.")
    parser.add_argument("--dropout",
                        default=0.3,
                        type=float)
    parser.add_argument("--num_epochs",
                        default=50,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--pretrain_num_epochs",
                        default=50,
                        type=int,
                        help="Total number of pretraining epochs to perform.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument("--device",
                        default='cuda:0',
                        type=str)
    parser.add_argument("--en",
                        action='store_true')
    parser.add_argument("--zh",
                        action='store_true')
    parser.add_argument("--fr",
                        action='store_true')
    parser.add_argument("--gloss",
                        action='store_true')
    parser.add_argument("--word",
                        action='store_true')
    parser.add_argument("--utils",
                        default='None',
                        type=str)

    return parser


def calculate_MAP_f1(output, indice, ground_truth, threshold):
    temp = []
    for i in range(len(ground_truth)):
        if ground_truth[i] == 1:
            temp.append(i)
    ground_truth = temp
    index = 1
    correct = 0
    MAP = 0
    for predicted_sememe in indice:
        if predicted_sememe in ground_truth:
            correct += 1
            MAP += (correct / index)
        index += 1
    MAP /= len(ground_truth)
    real_prediction = []
    for i in range(len(output)):
        if output[i] > threshold:
            real_prediction.append(i)
    prediction = real_prediction
    if len(list(set(prediction) & set(ground_truth))) == 0:
        f1 = 0
    else:
        recall = len(list(set(prediction) & set(ground_truth))) / \
            len(ground_truth)
        precision = len(list(set(prediction) & set(
            ground_truth))) / len(prediction)
        f1 = 2*recall*precision/(recall + precision)
    return MAP, f1


def get_model_name(args):
    model_name = []
    model_name.append(args.training_mode)
    model_name.append(args.pretrain_model)
    model_name.append(args.data_set)
    if args.load_model != 'NOT_LOAD':
        model_name.append('pretrained')
    model_name.append(args.data_feature)
    if args.hidden_size != 768 or args.img_hidden_size != 128:
        model_name.extend([str(args.hidden_size), str(args.img_hidden_size)])
    model_name.extend([str(args.encoder_learning_rate), str(args.classifier_learning_rate), str(args.dropout)])
    if args.img_classifier_learning_rate != 1e-3:
        model_name.append(str(args.img_classifier_learning_rate))
    if args.utils != 'None':
        model_name.append(args.utils)
    return '_'.join(model_name)+'.pt'


def multi_source_evaluate(mode, model, dataloader, device):
    if mode == 'pretrain':
        model.eval()
        all_MAP = 0.0
        all_f1 = 0.0
        all_loss = 0.0
        for ids, masks, labels, mask_idx in tqdm(dataloader):
            ids = ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            mask_idx = mask_idx.to(device)
            with torch.no_grad():
                loss, output, indice = model(mode='pretrain',
                                             text_ids=ids, text_mask=masks, labels=labels, mask_idx=mask_idx)
            all_loss += loss.item()
            output = output.detach().cpu().numpy().tolist()
            indice = indice.detach().cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            for i in range(len(output)):
                MAP, f1 = calculate_MAP_f1(
                    output[i], indice[i], labels[i], 0.3)
                all_MAP += MAP/len(output)
                all_f1 += f1/len(output)
        all_loss /= len(dataloader)
        all_MAP /= len(dataloader)
        all_f1 /= len(dataloader)
        return all_loss, all_MAP, all_f1
    elif mode == 'attention_multi_source':
        model.eval()
        result_list = {}
        all_MAP = 0.0
        all_f1 = 0.0
        all_loss = 0.0
        for idx, text_ids, text_mask, img_ids, img_mask, labels in tqdm(dataloader):
            text_ids = text_ids.to(device)
            text_mask = text_mask.to(device)
            img_ids = img_ids.to(device)
            img_mask = img_mask.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                loss, output, indice = model(mode=mode,
                                             text_ids=text_ids, text_mask=text_mask, img_ids=img_ids, img_mask=img_mask, labels=labels)
            all_loss += loss.item()
            output = output.detach().cpu().numpy().tolist()
            indice = indice.detach().cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            for i in range(len(output)):
                MAP, f1 = calculate_MAP_f1(
                    output[i], indice[i], labels[i], -1)
                result_list[idx[i]] = [MAP,f1,text_ids[i],labels[i],indice[i],output[i]]
                all_MAP += MAP/len(output)
                all_f1 += f1/len(output)
        all_loss /= len(dataloader)
        all_MAP /= len(dataloader)
        all_f1 /= len(dataloader)
        pickle.dump(result_list,open('result/main_valid','wb'))
        return all_loss, all_MAP, all_f1
    else:
        model.eval()
        # result_list = {}
        all_MAP = 0.0
        all_f1 = 0.0
        all_loss = 0.0
        for text_ids, text_mask, img_ids, labels in tqdm(dataloader):
            text_ids = text_ids.to(device)
            text_mask = text_mask.to(device)
            img_ids = img_ids.to(device)

            labels = labels.to(device)
            with torch.no_grad():
                loss, output, indice = model(mode=mode,
                                             text_ids=text_ids, text_mask=text_mask, img_ids=img_ids, labels=labels)
            all_loss += loss.item()
            output = output.detach().cpu().numpy().tolist()
            indice = indice.detach().cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            for i in range(len(output)):
                MAP, f1 = calculate_MAP_f1(
                    output[i], indice[i], labels[i], -1)
                # result_list[idx[i]] = [MAP,f1,text_ids[i],labels[i],indice[i],output[i]]
                all_MAP += MAP/len(output)
                all_f1 += f1/len(output)
        all_loss /= len(dataloader)
        all_MAP /= len(dataloader)
        all_f1 /= len(dataloader)
        # pickle.dump(result_list,open('result/main_valid','wb'))
        return all_loss, all_MAP, all_f1
