from collections import defaultdict
import os
import csv

from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
import xml.dom.minidom

from numpy import save


save_root2 = "GTSRB_train_xmls"


if not os.path.exists(save_root2):
    os.mkdir(save_root2)

dict = {'2': 'limitare_50',
        '7': 'limitare_100',
        '14': 'stop',
        '17': 'interzis_ambele_sensuri',
        '33': 'dreapta',
        '34': 'stanga',
        '35': 'inainte'}

for folder in range (0,42):
   os.mkdir("./" + save_root2 + "/Train/" + str(folder) )

def write_xml(folder, filename, bbox_list):
    e_width, e_height, e_xmin, e_ymin, e_xmax, e_ymax, e_class_name, e_filename = bbox_list[0]
    
    root = Element('annotation')
    SubElement(root, 'folder').text = folder
    if int(e_class_name) < 10 :
        SubElement(root, 'filename').text = filename[8:]
    if int(e_class_name) > 9 :
        SubElement(root, 'filename').text = filename[9:]
    #SubElement(root, 'path').text = './images' +  filename
    #source = SubElement(root, 'source')
    #SubElement(source, 'database').text = 'Unknown'

    # Details from first entry
    e_width, e_height, e_xmin, e_ymin, e_xmax, e_ymax, e_class_name, e_filename = bbox_list[0]
    
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = e_width
    SubElement(size, 'height').text = e_height
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'

    for entry in bbox_list:
        e_width, e_height, e_xmin, e_ymin, e_xmax, e_ymax, e_class_name, e_filename = entry
        
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = dict[e_class_name]
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = e_xmin
        SubElement(bbox, 'ymin').text = e_ymin
        SubElement(bbox, 'xmax').text = e_xmax
        SubElement(bbox, 'ymax').text = e_ymax

    #indent(root)

    
    #xml_str = root.toprettyxml(indent ="\t") 
    tree = ET.tostring(root)
    dom =  xml.dom.minidom.parseString(tree)
    
    pretty_xml_as_string = dom.toprettyxml()

    #print(pretty_xml_as_string)
 #   if int(e_class_name) < 10 :
 #       xml_filename = os.path.join('.', folder, os.path.splitext(filename[8:])[0] + '.xml')
 #   if int(e_class_name) > 9 :
 #       xml_filename = os.path.join('.', folder, os.path.splitext(filename[9:])[0] + '.xml')

    xml_filename = os.path.join('.', folder, os.path.splitext(filename)[0] + '.xml')

    #tree.write(xml_filename)
    with open(xml_filename, "w") as f:
        f.write(pretty_xml_as_string)
    
    f = open( xml_filename, 'r' )
    lines = f.readlines()
    f.close()

    f = open( xml_filename, 'w' )
    f.write('\n')
    f.write( ''.join( lines[1:] ) )
    f.close()
    

entries_by_filename = defaultdict(list)
print (entries_by_filename)
categories=["2","7","14","17","33","34","35"]

with open('Train.csv', 'r', encoding='utf-8') as f_input_csv:
    
    csv_input = csv.reader(f_input_csv)
    header = next(csv_input)

    for row in csv_input:
        width, height, x1, y1, x2, y2, category, file_name = row
        
        if category in categories:
            
            entries_by_filename[file_name].append(row)

for file_name, entries in entries_by_filename.items():
    print(file_name, len(entries))
    write_xml(save_root2, file_name, entries)