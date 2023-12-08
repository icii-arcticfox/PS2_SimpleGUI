#type:ignore
from Icii import *

class GUIButton(PythonAutomation): 

    _buttonCount = 0

    def ApplyAutomation(self):

        command = self.CodeScope.Name
        commandStyled = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(command)).title()

        with self.CodeAfterNext:
            plot_button = Button(master = window,
                command = ((command)),
                height = 2,
                width = ((len(commandStyled) + 2)),
                text = '((commandStyled))' )
            plot_button.grid(row=0, column=((GUIButton._buttonCount)), padx=5)
            window.columnconfigure( ((GUIButton._buttonCount)), weight=1)
        
        GUIButton._buttonCount += 1
