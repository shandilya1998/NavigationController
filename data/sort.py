import os
import argparse
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create train.txt and test.txt for backbone pretrai  ning")
    parser.add_argument('--datapath', type=str, help='data directory path')
    parser.add_argument('--split', type = int, default = 90, choices = list(range(1, 100)))
    args = parser.parse_args()
    lst = os.listdir(args.datapath)
    lst = [os.path.join(args.datapath, f) for f in lst if 'png' in f]
    split = int(len(lst) * args.split / 100)
    train_lst = lst[:split]
    test_lst = lst[split:]
    trainf = open(os.path.join(args.datapath, 'train.txt'), 'w')
    testf = open(os.path.join(args.datapath, 'test.txt'), 'w')
  
    for f in train_lst:
        trainf.write(f + '\n')
    trainf.close()
  
    for f in test_lst:
        testf.write(f + '\n')
    testf.close()
