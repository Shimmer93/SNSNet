import pickle
import argparse

def merge_pkl(orig_pkl, jbf_pkl, out_pkl):
    with open(orig_pkl, 'rb') as f:
        orig = pickle.load(f)
    with open(jbf_pkl, 'rb') as f:
        jbf = pickle.load(f)
    orig['annotations'] = jbf
    with open(out_pkl, 'wb') as f:
        pickle.dump(orig, f)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--orig_pkl', type=str)
    args.add_argument('--jbf_pkl', type=str)
    args.add_argument('--out_pkl', type=str)

    args = args.parse_args()
    merge_pkl(args.orig_pkl, args.jbf_pkl, args.out_pkl)