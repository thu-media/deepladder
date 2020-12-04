import os
for p in os.listdir('./ssim/'):
    f = open('./ssim/' + str(p), 'r')
    idx = 0
    for line in f:
        idx += 1
    f.close()
    if idx < 6:
        print(p)
        os.system('rm -rf ./feature/' + p + '.h5')
        os.system('rm -rf ./size/' + p)
        os.system('rm -rf ./ssim/' + p)
