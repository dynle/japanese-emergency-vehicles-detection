import os

dir = "./dataset/project-5-at-2022-12-21-08-38-9ce678bc/images/train"

root, dirs, files = next(os.walk(dir, topdown=True))
files = [ os.path.join(root, f) for f in files ]
# print(files)


with open('./dataset/project-5-at-2022-12-21-08-38-9ce678bc/train.txt', 'w') as f:
    for line in files:
        f.write(f"{line}\n")