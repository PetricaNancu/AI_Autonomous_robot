import os
import shutil
import xml.etree.ElementTree as ET


folder_path = ('./4_IDS/annotations')
target = ('./4_IDS/new_stop_xml/')

files = os.listdir(folder_path)

counter = 0
for file in files:

     if file.endswith(".xml"):
    
        xmlTree = ET.parse(folder_path + "/" + file)
        rootElement = xmlTree.getroot()
       
        for elem in rootElement.iter():
            for child in list(elem):
                if child.tag == 'name':
                    if child.text == 'stop':
                        #print("gasit")
                        counter+=1
                        if rootElement[4][0].text != "stop":
                            print("file :" + file + " - " +rootElement[4][0].text + " - " + str(counter))

                            shutil.copyfile(folder_path + "/" + file, target + file)

            



   