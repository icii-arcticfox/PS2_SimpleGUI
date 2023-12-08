#type:ignore
from Icii import *

class AddPlotToWindow(PythonAutomation): 
    def ApplyAutomation(self):

        visualize : Visualize = self.SelectFrom('Visualize', self.CodeScope)
        if visualize != None:
            visualize.CommentOutPltShow()

        histogram : Histogram = self.SelectFrom('Histogram', self.CodeScope)
        if histogram != None:
            histogram.CommentOutPltShow()

        with self.CodeScopeStart:
            plt.figure()
            
        with self.CodeMagix:
            fig = plt.gcf()
            canvas = FigureCanvasTkAgg(fig, master=window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=1, column=0, columnspan=window.grid_size()[0], pady=10)

