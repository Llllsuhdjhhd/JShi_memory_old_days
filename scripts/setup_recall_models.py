"""Download the native local embedding and official quantized ranking weights."""
from huggingface_hub import list_repo_files, snapshot_download
import argparse

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true',help='Also cache original PyTorch reranker weights')
    args=parser.parse_args()
    embedding='BAAI/bge-small-zh-v1.5'
    files=set(list_repo_files(embedding))
    weight='model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
    print(snapshot_download(embedding,allow_patterns=[weight,'*.json','*.txt','*.model','1_Pooling/*']))
    ranker='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
    patterns=['*.json','*.txt','*.model','onnx/model_quint8_avx2.onnx']
    if args.gpu:
        ranker_files=set(list_repo_files(ranker))
        patterns.append('model.safetensors' if 'model.safetensors' in ranker_files else 'pytorch_model.bin')
    print(snapshot_download(ranker,allow_patterns=patterns))
