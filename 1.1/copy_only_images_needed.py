import os
import shutil
import random

categories=["2","7","14","17","33","34","35"]


#folder_xmls = ('./GTSRB_train_xmls/Train/')
#folder_images = ('./GTSRB_old_Merged')
#target = ('./GTSRB_train_xmls/Train/Train')


folder_xmls = ('./GTSRB_train_xmls/Test/')
folder_images = ('./GTSRB_test_images')
target = ('./GTSRB_train_xmls/Test/Test')

for sign in categories:
    path_xml = folder_xmls + sign

    images = os.listdir(folder_images)
    xmls = os.listdir(path_xml)


    list1 = []
    list_selected = []
    for x in xmls:

        list1.append(x)

    dim=len(list1)
    copy = 100
    if dim > copy:
        list_selected = random.sample(list1, copy)
        print(list_selected)
    else:
        for entry in xmls:
            list_selected.append(entry)
        print ( " not enough samples ( lower than "+  str(copy) + ")")


    for entry in list_selected:

        
        aux = "/" + entry[0:-3]+"png"
        
        print (aux )
        shutil.copyfile(folder_images + aux, target + aux)
        shutil.copyfile(path_xml + "/" + entry, target + "/" + entry)
