import numpy as np
import xml.etree.ElementTree as ET

def parse_bboxes(ann, classes):

    """Parse the annotations xml file

    Parameters
    ----------
    ann: xml ET parser
        The file containing the bounding boxes


    Returns
    -------

    numpy.ndaaray
        Parsed bounding box co-ordinates of the format `n x 5` where n is
        number of bounding boxes and 5 represents `x1,y1,x2,y2,c` of the box

    """

    names, xmins, ymins, xmaxs, ymaxs = [], [], [], [], []
    ann_root = ann.getroot()

    for name in ann_root.iter('name'):
        names.append(np.float32(classes.index(name.text)))

    for xmin in ann_root.iter('xmin'):
        xmins.append(np.float32(xmin.text))

    for ymin in ann_root.iter('ymin'):
        ymins.append(np.float32(ymin.text))

    for xmax in ann_root.iter('xmax'):
        xmaxs.append(np.float32(xmax.text))

    for ymax in ann_root.iter('ymax'):
        ymaxs.append(np.float32(ymax.text))

    return np.column_stack((xmins, ymins, xmaxs, ymaxs, names))


def encode_bboxes(ann, bboxes, img_name):

    """Encode bboxes in a new xml file

            Parameters
            ----------
            ann: xml ET parser
                The file containing the bounding boxes


            Returns
            -------

            numpy.ndaaray
                Parsed bounding box co-ordinates of the format `n x 5` where n is
                number of bounding boxes and 5 represents `x1,y1,x2,y2,c` of the box

            """

    ann_root = ann.getroot()

    folder = ET.Element("folder")
    folder.text = ann_root.find('folder').text
    filename = ET.Element("filename")
    filename.text = img_name
    path = ET.Element("path")
    path.text = ann_root.find('folder').text + '/' + img_name
    source = ET.Element("source")
    database = ET.Element("database")
    database.text = ann_root.find("source").find('database').text
    source.append(database)
    size = ET.Element("size")
    width = ET.Element("width")
    width.text = ann_root.find("size").find('width').text
    height = ET.Element("height")
    height.text = ann_root.find("size").find('height').text
    depth = ET.Element("depth")
    depth.text = ann_root.find("size").find('depth').text
    size.append(width)
    size.append(height)
    size.append(depth)
    segmented = ET.Element("segmented")
    segmented.text = ann_root.find('segmented').text

    new_root = ET.Element("annotation")
    new_root.append(folder)
    new_root.append(filename)
    new_root.append(path)
    new_root.append(source)
    new_root.append(size)
    new_root.append(segmented)

    for b in bboxes:
        xmin = ET.Element("xmin")
        xmin.text = str(int(b[0]))
        ymin = ET.Element("ymin")
        ymin.text = str(int(b[1]))
        xmax = ET.Element("xmax")
        xmax.text = str(int(b[2]))
        ymax = ET.Element("ymax")
        ymax.text = str(int(b[3]))
        name = ET.Element("name")
        name.text = self.classes[int(b[4])]
        bndbox = ET.Element("bndbox")
        bndbox.append(xmin)
        bndbox.append(ymin)
        bndbox.append(xmax)
        bndbox.append(ymax)
        pose = ET.Element("pose")
        truncated = ET.Element("truncated")
        difficult = ET.Element("difficult")
        pose.text = "Unspecified"
        truncated.text = "0"
        difficult.text = "0"
        obj = ET.Element("object")
        obj.append(name)
        obj.append(pose)
        obj.append(truncated)
        obj.append(difficult)
        obj.append(bndbox)

        new_root.append(obj)

    new_tree = ET.ElementTree(new_root)

    return new_tree