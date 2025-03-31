import os
import pickle
from datetime import *

import platform
import numpy as np
import tensorflow as tf
from util.load_hep import Loader, INFO_LOG

from Config import Config
from util.model_saver import DynGCN_saver, DynGCN_loader
from util.LearningRateUpdater import LearningRateUpdater
from model.dynGCN import DynGCN
# from model.dynGCN import DynGCN
from util.evalutate import F1score
import time

# 是在给定的会话（session）中运行模型（model），并根据模型的模式（Train 或 Valid）进行训练或验证操作，同时计算并返回相关的性能指标和损失值
# session：TensorFlow 的会话对象，用于执行计算图中的操作。
# config：配置对象，可能包含模型的各种超参数和设置。
# model：要运行的模型对象，包含模型的结构、操作和相关属性。
# loader：数据加载器对象，用于加载数据和生成批次数据。
# verbose：布尔值，用于控制是否打印详细的中间结果信息，默认为 False。
def run(session, config, model, loader, verbose=False):
    # 记录总损失（total_cost）、处理的批次数量（num_）、AUC 值（auc）、F1 分数计算对象（f1_score）
    # 预测结果列表（prediction_l 和 prediction_n_l）以及临时的 AUC 计算变量（t_auc 和 t_num）
    total_cost = 0.

    num_ = 0.

    auc = 0.
    f1_score = F1score(model.batch_size)
    prediction_l = [0.] * model.batch_size
    prediction_n_l = [0.] * model.batch_size
    t_auc = 0.
    t_num = 0.
    #内部辅助函数 _add_list，用于将两个列表对应位置的元素相加，并返回结果列表。
    def _add_list(x, y):
        for idx in range(len(x)):
            x[idx] += y[idx]
        return x
    # 初始化时间消耗变量（time_consume），并从数据加载器 loader 中
    # 获取上一个时间步的嵌入特征（feature_h0）以及当前图的邻接矩阵（adj_now）和邻接矩阵的变化（delta_adj）。
    time_consume = 0.
    feature_h0 = loader.last_embeddings()
    adj_now, delta_adj = loader.adj()

    #遍历由数据加载器 loader 生成的批次数据，根据模型的模式（Train 或 Valid）和批次大小（model.batch_size）获取每个批次的数据
    for batch in loader.generate_batch_data(batchsize=model.batch_size, mode=model.mode):

        batch_id, batch_num, nodelist1, nodelist2, negative_list = batch
        #构建输入数据字典 feed，包含节点列表、邻接矩阵等数据，指定要运行的操作列表 out（包括损失、优化器、AUC 结果等）
        #使用会话 session 运行这些操作并获取结果。将预测结果累加到相应的列表中。
        if model.mode == "Train":
            feed = {
                model.input_x: nodelist1,
                model.input_y: nodelist2,
                model.adj_now: adj_now, #将adj_now赋给对应的占位符
                model.delta_adj: delta_adj,
                model.feature_h0:feature_h0,
                model.negative_sample: negative_list
            }
            print('!!!!!!!!!!!!!!!!!!!!')
            print(f'model.input_x:{model.input_x}')
            print(f'model.input_y:{model.input_y}')
            print(f'model.adj_now:{model.adj_now}')
            print(f'model.delta_adj:{model.delta_adj}')
            print(f'model.feature_h0:{model.feature_h0}')
            print(f'model.negative_sample:{model.negative_sample}')
            print('*************************')
            out = [model.cost, model.optimizer, model.auc_result,
                   model.auc_opt, model.prediction, model.prediction_n,
                   model.test1, model.test2
                   ]
            #这里出问题
            output = session.run(out, feed)
            cost, _, auc, _, prediction, prediction_n, test1, test2 = output

            prediction_l = _add_list(prediction_l, prediction)
            prediction_n_l = _add_list(prediction_n_l, prediction_n)
        #当模型处于验证模式时，构建不同的输入数据字典 feed（包含调整后的节点列表和标签）
        #指定要运行的操作列表 out，记录开始时间，运行操作并计算时间消耗。根据预测结果计算临时的 AUC 值。
        if model.mode == "Valid":
            # print "nodelist1", np.asarray(nodelist1 * 2).shape
            # print np.asarray(nodelist2 + negative_list).shape
            # print np.asarray([1] * (model.batch_size / 2) + [0] * (model.batch_size /2)).shape

            feed = {
                model.input_x: np.asarray(nodelist1 * 2),
                model.input_y: np.asarray(nodelist2 + negative_list),
                model.adj_now: adj_now,
                model.delta_adj: delta_adj,
                model.feature_h0:feature_h0,
                model.label_xy: np.asarray([1] * (model.batch_size / 2) + [0] * (model.batch_size /2))
            }

            out = [model.cost, model.optimizer, model.auc_result,
                   model.auc_opt, model.prediction]
            begin_time = time.time()
            output = session.run(out, feed)
            time_consume += time.time() - begin_time
            # print output
            cost, _, auc, _, prediction = output

            for idx in range(len(prediction) / 2):
                if prediction[idx] > prediction[idx + len(prediction) / 2]:
                    t_auc += 1
                t_num += 1
            # print prediction
        # print "TEST",prediction

        #根据模型模式处理结果，训练模式下累加损失，验证模式下计算 F1 分数。
        #如果 verbose 为 True 且满足一定条件（batch_id % int(batch_num / 5.) == 1），则打印详细的中间结果信息。
        if model.mode == "Train":
            auc = 0.
            total_cost += cost
        else:
            f1_score.add_f1(
                np.asarray([1] * (model.batch_size / 2) + [0] * (model.batch_size / 2)), prediction
            )
            cost = 0.
            total_cost += cost

        num_ += 1.
        if verbose and batch_id % int(batch_num / 5.) == 1 and model.mode == "Valid":
            INFO_LOG("{}/{}, cost: {}, auc: {}, f1_score: {}".format(
                batch_id, batch_num, total_cost / num_,
                auc, f1_score.return_f1_score()
            ),
            True
            )

    #如果处理的批次数量为 0，则打印错误信息。在验证模式下计算最终的 AUC 值。最后返回平均损失和包含 AUC、F1 分数以及时间消耗的性能指标字典。
    if num_ == 0:
        INFO_LOG("===failed graph===" + str(loader.present_graph), True)
    # if model.mode == "Train":
    #     print("prediction_l", [x / batch_num for x in prediction_l])
    #     print("prediction_l_n", [x / batch_num for x in prediction_n_l])
    # else:
    #     print("valid prediction", f1_score.return_predict_mean())
    if not model.mode == "Train":
        # print "auc", t_auc / t_num
        auc = t_auc / t_num

    return total_cost / num_, {"auc": auc, "f1_score": f1_score.return_f1_score(), "time_consume": time_consume}


def main(_):
    # 程序的main函数，程序进来从这里开始运行
    loader = Loader(flag="as") # 实例化Loader,Loader 类主要用于加载和处理与图数据相关的操作，
                               # 包括加载图数据、计算图的变化、生成批次数据
    config = Config(loader, flag="as")  # 实例化Config,主要用于存储和管理与某种模型训练或任务相关的配置参数，
                                       # 并根据传入的 loader 对象获取一些动态的配置信息

    if platform.system() == 'Linux':
        gpuid = config.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpuid)
        device = '/gpu:' + str(gpuid)
    else:
        device = '/cpu:0' # 指定后续的计算操作将在哪个设备上执行 0 通常代表第一个设备

    lr_updater = LearningRateUpdater(config.learning_rate, config.decay, config.decay_epoch)
    #创建一个学习率更新器对象 lr_updater，用于在模型训练过程中动态调整学习率
    #主要作用是在模型训练过程中动态更新学习率，以帮助模型更好地收敛

    i = 0
    graph = tf.Graph()
    with graph.as_default():
        trainm = DynGCN(config, device, loader, "Train") #构筑了用于训练和验证的模型结构
        testm = DynGCN(config, device, loader, "Valid")

    #配置session参数
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=session_config) as session:#创建会话
        # print "!!!!!!!!!!!"
        session.run(tf.global_variables_initializer())# 在会话中运行全局变量初始化操作，将计算图中定义的所有全局变量初始化为它们的默认值
                                                      # 为模型训练做好准备。
        # print "*********"
        # trainm.load_last_time_embedding(loader.present_graph, session)

        # CTR_GNN_loader(session, config)

        #初始化一些变量用于记录模型在训练过程中的最优性能指标（best_f1_score 和 best_auc_score）、
        #对应的最优轮次（best_epoch）以及时间消耗相关的统计信息（time_consume_t 和 sum_time_consume）。
        #不过最后两个似乎没有用上
        best_f1_score = 0.
        best_auc_score = 0.
        best_epoch = 0
        time_consume_t = 0.
        sum_time_consume = 0.
        for epoch in range(config.epoch_num):# epoch_num是100000
            trainm.update_lr(session, lr_updater.get_lr())#调用 trainm 模型的 update_lr 方法，根据 lr_updater 获取到的当前学习率
                                                          # 在会话中更新训练模型的学习率。
            # session.run(tf.local_variables_initializer())
            # #
            # 在会话中运行训练模型 trainm，传入会话、配置、模型、数据加载器等参数，获取训练的损失值 cost 和评估结果 eavluation_result。
            # 然后使用 INFO_LOG 函数（具体实现未给出）记录训练的相关信息，包括评估结果和每 100 轮的训练损失
            # 这里出问题
            cost, eavluation_result = run(session, config, trainm, loader, verbose=False)
            INFO_LOG("Epoch %d  Train " % epoch + str(eavluation_result), epoch % 1 == 0)
            INFO_LOG("Epoch %d Train costs %.3f" %
                     (epoch, cost), epoch % 100 == 0)

            #局部状态的重置，比如在不同轮次或不同操作之间的变量初始化。
            session.run(tf.local_variables_initializer())

            #调用 run 函数在会话中运行验证模型 testm，获取验证的损失值和评估结果，并记录验证的相关信息。
            cost, eavluation_result = run(session, config, testm, loader, verbose=False)
            INFO_LOG("Epoch %d  Valid " % epoch + str(eavluation_result), epoch % 1 == 0)
            INFO_LOG("Epoch %d Valid cost %.3f" % (epoch, cost), epoch % 1 == 0)
            # #
            #从验证结果中提取 AUC 值和微平均 F1 分数（micro_f1_score），然后调用 lr_updater 的 update 方法，
            # 根据当前的 F1 分数和轮次来更新学习率
            auc = eavluation_result['auc']
            f1_score = eavluation_result["f1_score"]["micro_f1_score"]
            lr_updater.update(f1_score, epoch)

            #比较当前的 F1 分数和 AUC 值与之前记录的最优值，如果当前值更好，则更新最优值、记录对应的轮次，
            # 并调用 DynGCN_saver 函数（具体实现未给出）保存当前最优的模型。同时，使用 INFO_LOG 函数记录相关的最优指标信息
            if best_f1_score < f1_score:
                best_f1_score = f1_score
                best_epoch = epoch
                DynGCN_saver(session, config, best_f1_score, best_epoch, "hep")

                INFO_LOG("*** best f1_score now is %.5f in %d epoch" % (best_f1_score, best_epoch), True)
                INFO_LOG("BEST Epoch %d  Valid " % epoch + str(eavluation_result), True)

            if best_auc_score < auc:
                best_auc_score = auc
                INFO_LOG("*** best auc now is %.5f in %d epoch" % (best_auc_score, epoch), True)

            INFO_LOG("*** best f1_score now is %.4f in %d epoch" % (best_f1_score, best_epoch), epoch % 100 == 0)
            # time_consume_t += 1.
            # sum_time_consume += eavluation_result["time_consume"]
            # print("TIME CONSUME *** ", sum_time_consume/ time_consume_t)

            #每训练一轮（epoch % 1 == 0）且不是第一轮（epoch != 0）时，调用数据加载器 loader 的 change_2_next_graph_date 方法，
            # 更新数据加载器的状态，可能是切换到下一个时间步或下一批数据
            if epoch % 1 == 0 and epoch != 0:
                loader.change_2_next_graph_date()



if __name__ == '__main__':  # 判断当前文件是否是作为脚本直接运行
    tf.app.run() # 解析命令行参数，然后调用程序中的main函数
