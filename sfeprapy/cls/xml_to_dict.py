import xml.etree.ElementTree as ET
from collections import defaultdict

__all__ = ['XML2Dict']


class XML2Dict(object):
    def __init__(self, coding='UTF-8'):
        self._coding = coding

    def _parse_node(self, t):
        d = {t.tag: {} if t.attrib else None}  # the variable 'd' is the constructed target dictionary
        # 't.tag' if have values, it is the first layer of the dictionary
        children = list(t)  # The following recursive traverse processing tree, until the leaf node
        if children:  # Determine whether the node is empty, recursive boundary conditions
            dd = defaultdict(list)
            for dc in map(self._parse_node, children):  # recursive traverse processing tree
                for k, v in dc.iteritems():
                    dd[k].append(v)
            d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.iteritems()}}  # handle child node
        if t.attrib:  # handle attributes,prefix all of the stored attributes @
            d[t.tag].update(('@' + k, v) for k, v in t.attrib.iteritems())
        if t.text:
            text = t.text.strip().encode(self._coding)  # strip blank space
            if children or t.attrib:
                d[t.tag]['#text'] = text
            else:
                d[t.tag] = text  # the text value as t.tag
        return d

    def parse(self, xml_file):
        with open(xml_file, 'r') as fp:
            return self.fromstring(fp.read())

    def fromstring(self, xml_str):
        t = ET.fromstring(xml_str)
        return self._parse_node(t)