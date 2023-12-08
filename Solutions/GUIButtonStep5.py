#type:ignore
from Icii import *

class GUIButton(PythonAutomation): 
    def ApplyAutomation(self):

        command = self.CodeScope.Name
        commandStyled = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(command)).title()

        with self.CodeAfterNext:
            plot_button = Button(master = window,
                command = ((command)),
                height = 2,
                width = ((len(commandStyled) + 2)),
                text = '((commandStyled))' )
            plot_button.grid(row=0, column=0, padx=5)
            window.columnconfigure( 0, weight=1)
