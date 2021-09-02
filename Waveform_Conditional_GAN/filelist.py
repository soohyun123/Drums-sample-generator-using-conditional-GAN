from os import listdir
from os.path import isfile, join

mypath = "/projects/c_drums/"

for name in listdir(mypath):

    onlywaves = [x for x in listdir(mypath+name) if x.endswith(".wav")]

    with open(mypath+name+"_list.txt", 'w') as f:
        for item in onlywaves:
            f.write(name+"/%s\n" % item)