import os

dir = "./dataset/images/train"

root, dirs, files = next(os.walk(dir, topdown=True))
files = [ os.path.join(root, f) for f in files ]
# print(files)


with open('./dataset/train.txt', 'w') as f:
    for line in files:
        f.write(f"{line}\n")