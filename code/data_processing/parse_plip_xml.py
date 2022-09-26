import xml.etree.ElementTree as ET
import sys
 
xml_file = sys.argv[1]

tree = ET.parse(xml_file)
root = tree.getroot()
print(root.tag)