python3 baselines.py --dataset cora --model_tag ladies_64_cora > ./summary.txt
mv ./summary.txt ../results/ladies_64_cora/summary.txt
python3 baselines.py --dataset citeseer --model_tag ladies_64_citeseer > ./summary.txt
mv ./summary.txt ../results/ladies_64_citeseer/summary.txt
python3 baselines.py --dataset pubmed --model_tag ladies_64_pubmed > ./summary.txt
mv ./summary.txt ../results/ladies_64_pubmed/summary.txt