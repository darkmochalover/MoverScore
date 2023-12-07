import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
 
 
import copy
import numpy as np
import sys
import torch

def test_metric(data):
    from eva.tokenizer import SimpleTokenizer, PretrainedTokenizer
    tokenizer = SimpleTokenizer(method="nltk")
    eva_path = "/home/guanjian/evaluation_dataset/metrics/OpenEva"

    metric_score = {}

    from eva.moverscore import MoverScore
    moverscore_metric = MoverScore()
    # print(moverscore_metric.info())
    metric_score.update(moverscore_metric.compute(data, batch_size=8))

    for metric in metric_score:
        for d, r in zip(data, metric_score[metric]):
            d["metric_score"][metric] = r
    return data

def test_heva(data, figure_path="./figure"):
    from eva.heva import Heva
    heva = Heva([1,2,3,4,5])
    print("consistency of human evaluation:", heva.consistency([d["score"] for d in data]))
    
    # human evaluation, consistency & distribution
    data_model = {}
    for d in data:
        if d["model_name"] in data_model:
            data_model[d["model_name"]].append(np.mean(d["score"]))
        else:
            data_model[d["model_name"]] = [np.mean(d["score"])]

    print("="*20)
    for model_name in data_model:
        heva.save_distribution_figure(score=data_model[model_name], save_path=figure_path, model_name=model_name, ymin=0, ymax=50)
        for model2_name in data_model:
            print("mean score of %s: %.4f, %s: %.4f; " % (model_name, np.mean(data_model[model_name]), model2_name, np.mean(data_model[model2_name])), end="\t")
            mean_test_result = heva.mean_test(data_model[model_name], data_model[model2_name])
            print("(mean testing) t-statistic=%.4f, p-value=%.4f"%(mean_test_result["t-statistic"], mean_test_result["p-value"]))
    # correlation between human evaluation and automatic evaluation
    metric_name_list = list(data[0]["metric_score"].keys())
    data_metric = {}
    for d in data:
        for metric_name in metric_name_list:
            if metric_name in data_metric:
                data_metric[metric_name].append(d["metric_score"][metric_name])
            else:
                data_metric[metric_name] = [d["metric_score"][metric_name]]
    human_score = np.mean([d["score"] for d in data], 1)
    corr = {}
    for metric_name in metric_name_list:
        heva.save_correlation_figure(human_score, data_metric[metric_name], save_path=figure_path, metric_name=metric_name)
        corr[metric_name] = heva.correlation(data_metric[metric_name], human_score)
    print("="*20)
    print(corr)

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    return device
        

if __name__ == "__main__":
    device = get_device() # "cuda:3", "cpu"
    print(device)
    # quit()
    

    # if "cuda" in device:
    #     gpu_name = device.split(":")[-1]
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_name
    #     print("using %s-th gpu"%gpu_name)
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ""
    #     print("using cpu")

    '''
    "한 번은 슈퍼마리오의 세계에서 놀던 백설공주가 있었습니다. 그녀는 화려한 성에서 함께 사는 일곱 난장이와 행복한 일상을 보내고 있었습니다. 어느 날, 마리오와 루이지는 백설공주의 세계로 떠나게 " \
        "되었습니다. 처음엔 낯선 세계에 어색함을 느낀 백설공주였지만, 함께 뛰어노는 슈퍼마리오와 루이지의 모습에 빨리 적응하게 되었습니다. 그녀도 슈퍼 파워를 얻어 매우 높이 점프하고 벽을 올라가며 " \
        "재미있게 놀았습니다. 하지만 한가운데 덕분에 걸린 악당이 나타났습니다. 그는 백설공주의 세계를 어지럽히고 평화를 무너뜨리려는 교활한 브라더스입니다. 백설공주와 슈퍼마리오는 힘을 합쳐 이 두 사람의 " \
        "악한 계획을 막기로 결심했습니다. 일곱 난장이들은 각자의 특별한 능력을 발휘하여 슈퍼마리오와 백설공주를 도왔습니다. 화려한 아이템과 슈퍼 파워의 조합으로 브라더스의 함정을 피하고 어려운 미션을 " \
        "해결해 나갔습니다. 계속된 모험 끝에 백설공주와 슈퍼마리오는 브라더스의 악한 계획을 무산시켰습니다. 세계는 다시 한 번 평화롭게 되었고, 백설공주는 일곱 난장이와 함께 새로운 친구들을 얻어 더욱 " \
        "풍요로운 일상을 즐기게 되었습니다. 마리오와 루이지는 돌아가기 전에 백설공주에게 슈퍼마리오 세계의 기술과 문화를 가르쳐주었습니다. 그리고 이제 두 세계의 사람들은 서로를 이해하고 협력하여 새로운 " \
        "세계를 만들어가게 되었습니다. 이후로 백설공주는 슈퍼마리오 세계와 자신의 세계를 오가며 친구들과 함께 모험을 즐기고, 브라더스와 같은 악당들을 상대로 싸우며 두 세계의 평화를 함께 지켜나갔습니다. " \
        "그녀의 용기와 친절한 마음은 어느 세계에서나 사람들에게 희망과 영감을 주었습니다."
    '''

    기 = "한 번은 슈퍼마리오의 세계에서 놀던 백설공주가 있었습니다. 그녀는 화려한 성에서 함께 사는 일곱 난장이와 행복한 일상을 보내고 있었습니다."
        
    승 = "어느 날, 마리오와 루이지는 백설공주의 세계로 떠나게 " \
        "되었습니다. 처음엔 낯선 세계에 어색함을 느낀 백설공주였지만, 함께 뛰어노는 슈퍼마리오와 루이지의 모습에 빨리 적응하게 되었습니다. 그녀도 슈퍼 파워를 얻어 매우 높이 점프하고 벽을 올라가며 " \
        "재미있게 놀았습니다."
        
    전 = "하지만 한가운데 덕분에 걸린 악당이 나타났습니다. 그는 백설공주의 세계를 어지럽히고 평화를 무너뜨리려는 교활한 브라더스입니다. 백설공주와 슈퍼마리오는 힘을 합쳐 이 두 사람의 " \
        "악한 계획을 막기로 결심했습니다. 일곱 난장이들은 각자의 특별한 능력을 발휘하여 슈퍼마리오와 백설공주를 도왔습니다. 화려한 아이템과 슈퍼 파워의 조합으로 브라더스의 함정을 피하고 어려운 미션을 " \
        "해결해 나갔습니다. 계속된 모험 끝에 백설공주와 슈퍼마리오는 브라더스의 악한 계획을 무산시켰습니다."
        
    결 = "세계는 다시 한 번 평화롭게 되었고, 백설공주는 일곱 난장이와 함께 새로운 친구들을 얻어 더욱 " \
        "풍요로운 일상을 즐기게 되었습니다. 마리오와 루이지는 돌아가기 전에 백설공주에게 슈퍼마리오 세계의 기술과 문화를 가르쳐주었습니다. 그리고 이제 두 세계의 사람들은 서로를 이해하고 협력하여 새로운 " \
        "세계를 만들어가게 되었습니다. 이후로 백설공주는 슈퍼마리오 세계와 자신의 세계를 오가며 친구들과 함께 모험을 즐기고, 브라더스와 같은 악당들을 상대로 싸우며 두 세계의 평화를 함께 지켜나갔습니다. " \
        "그녀의 용기와 친절한 마음은 어느 세계에서나 사람들에게 희망과 영감을 주었습니다."


    # data = [
    #         {
    #             'context': "Jian is a student.",
    #             'reference': ["Jian comes from Tsinghua University."],
    #             'candidate': "what the fuck",
    #             'model_name': "gpt",
    #             'score': [5, 5, 5],
    #             'metric_score': {},
    #         },
    #         {
    #             'context': "Jian is a worker.",
    #             'reference': ["Jian came from China. Jian was running."],
    #             'candidate': "He came from China.",
    #             'model_name': "gpt",
    #             'score': [4, 4, 4],
    #             'metric_score': {},
    #         }
    #     ]
    
    data = [
            {
                'context': 승,
                'reference': [기],
                'candidate': 전,
                'model_name': "gpt",
                'score': [5, 5, 5],
                'metric_score': {},
            },
            {
                'context': 전,
                'reference': [승],
                'candidate': 결,
                'model_name': "gpt",
                'score': [4, 4, 4],
                'metric_score': {},
            }
        ]

    figure_path="./figure"
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    data_metric = test_metric(data)
    print('metric = ', data_metric)
    test_heva(data_metric)