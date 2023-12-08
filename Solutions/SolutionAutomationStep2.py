#type:ignore
from Icii import *

class GUI(PythonAutomation): 
    def ApplyAutomation(self):

        with self.CodeImport:
            from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
            from tkinter import *

        with self.CodeAfterAutomation:
            window = Tk()
            window.title( 'Training Plots' )
            window.geometry("800x600")
            
        with self.CodeScriptEnd:
            window.mainloop()
