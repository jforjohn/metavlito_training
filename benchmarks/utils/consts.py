
class Split:
    TRAIN = 'train'
    VAL = 'validation'
    TEST = 'test'


class CsvField:
    OBJECT_ID = 0
    TARGET = 1
    IMAGE_FILE = 2
    WIDTH = 3
    HEIGHT = 4
    PRODUCT_SIZE = 5
    ASPECT_RATIO = 6
    SPLIT = 7

class CsvFieldLiteral:
    OBJECT_ID = 'ID'
    TARGET = 'Label'
    IMAGE_FILE = 'Image file'
    SPLIT = 'Split'
    DATA = 'Width,Height'