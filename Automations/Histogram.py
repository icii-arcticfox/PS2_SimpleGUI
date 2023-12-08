#type:ignore
from Icii import *
import re

class Histogram(PythonAutomation): 

    def GetAutomationDescription(self):
        return 'Graphs a histogram for the given data / variable'

    def GetItemDescriptions(self):
        return [
            ('value', 'The variable, should be list like, to graph as a histogram', ItemType.Normal, ItemTypeHelp.String)
        ]
    
    def ApplyAutomation(self):

        value = str(self.Items.Get('value', 0))
        name = value

        pattern = r'\w+\s*\[\s*[\"\']([^\"\']+)[\"\']\s*\]'
        match = re.findall(pattern, name)
        if len(match) > 0:
            name = match[0]

        with self.CodeImport:
            import matplotlib.pyplot as plt
            import numpy as np

        with self.CodeMagix:
            ((name))Counts, ((name))Bins = np.histogram(((value)))
            plt.title('((name)) Histogram')
            plt.xlabel('((value))')
            plt.ylabel('counts')
            plt.stairs(((name))Counts, ((name))Bins)
            ((self.NewMarker('HistogramPossibleCommentOut')))plt.show()

    def CommentOutPltShow(self):
        self.CodeScope.ReplaceMarker('HistogramPossibleCommentOut', "# plt");

