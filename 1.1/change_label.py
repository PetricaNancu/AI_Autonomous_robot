import os
import shutil
import xml.etree.ElementTree as ET
import random

folder_xmls = ('./GTSRB/images/34_selected_xml')

target = ('./GTSRB/images/34_selected_xml/')

xmls = os.listdir(folder_xmls)

for file in xmls:

    if file.endswith(".xml"):
    
        xmlTree = ET.parse(folder_xmls + "/" + file)
        rootElement = xmlTree.getroot()
       
        rootElement.find('filename').text = file[:-3] + "png"
        #Write the modified xml file.        

        #for element in rootElement.findall("object"):
    
        #    element.find('name').text = "dreapta"
            #Write the modified xml file.        
        xmlTree.write(target + file)
        f = open(folder_xmls+"/"+ file, 'r' )
        lines = f.readlines()
        f.close()

        f = open( target +"/"+file, 'w' )
        f.write('\n')
        f.write( ''.join( lines[:] ) )
        f.close()
        