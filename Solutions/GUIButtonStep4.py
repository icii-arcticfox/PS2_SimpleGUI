#type:ignore
from Icii import *

class GUIButton(PythonAutomation): 
    def ApplyAutomation(self):

        with self.CodeAfterNext:
            plot_button = Button(master = window,
                command = showTrainingImage,
                height = 2,
                width = 19,
                text = 'Show Training Image' )
            plot_button.grid(row=0, column=0, padx=5)
            window.columnconfigure( 0, weight=1)
