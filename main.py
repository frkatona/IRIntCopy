from tkinter import *
import tkinter as tk
from tkinter import filedialog
import backend
import pathlib

class MyWindow:
    def __init__(self, win, x1, x2, n, y1):

        # Labels
        self.lbl_imp = Label(win, text = 'Select Import Directory')
        self.lbl_exp = Label(win, text = 'Select Export Directory')
        self.lbl_controls = Label(win, text = '# Controls')
        self.lbl_intwin = Label(win, text = 'Integration Windows')

        # Entry Fields
        self.controlsField = Entry()

        # Checkboxes
        cb_spectra_isChecked = tk.IntVar(0)
        cb_peak_isChecked = tk.IntVar(0)
        cb_1_isChecked = tk.IntVar(0)
        cb_2_isChecked = tk.IntVar(0)
        cb_3_isChecked = tk.IntVar(0)
        cb_4_isChecked = tk.IntVar(0)
        cb_5_isChecked = tk.IntVar(0)
        self.cb_spectra = Checkbutton(win, text = 'Spectra (Line)', variable = cb_spectra_isChecked, command = self.consolelog)
        self.cb_peak = Checkbutton(win, text = 'Peak Integrations (Bar)', variable = cb_peak_isChecked)
        self.cb_1 = Checkbutton(win, text = 'Si-O-Si (715 - 830)', variable = cb_1_isChecked) 
        self.cb_2 = Checkbutton(win, text = 'Si-O-Si (940 - 1230)', variable = cb_2_isChecked) 
        self.cb_3 = Checkbutton(win, text = 'Si-H (2290 - 2390)', variable = cb_3_isChecked) 
        self.cb_4 = Checkbutton(win, text = 'CH3 (2900 - 2970)', variable = cb_4_isChecked) 
        self.cb_5 = Checkbutton(win, text = 'Vinyl (3060 - 3080)', variable = cb_5_isChecked) 

        # Buttons
        self.btn_selectImport = Button(win, text = '...', command = self.selectImportPath)
        self.btn_selectExport = Button(win, text = '...', command = self.selectExportPath)
        self.btn_graph = Button(win, text = 'Generate Graphs', command = self.graph)
        self.btn_exp = Button(win, text = 'Export to CSV') 

        # Formatting
        
        # DIRECTORY SELECTORS #
        self.lbl_imp.place(x = x1, y = 20)
        self.btn_selectImport.place(x = x1 + 135, y = 20)

        self.lbl_exp.place(x = x1, y = 420)
        self.btn_selectExport.place(x = x1 + 135, y = 420)

        
        # GRAPH OPTIONS #
        self.cb_spectra.place(x = x1, y = y1)
        self.cb_peak.place(x = x2, y = y1)

        self.lbl_controls.place(x = x2 + n, y = y1 + n)
        self.controlsField.place(x = x2 + n + 65, y = y1 + n)

        self.cb_1.place(x = x2 + n, y = y1 + (n*2))
        self.cb_2.place(x = x2 + n, y = y1 + (n*3))
        self.cb_3.place(x = x2 + n, y = y1 + (n*4))
        self.cb_4.place(x = x2 + n, y = y1 + (n*5))
        self.cb_5.place(x = x2 + n, y = y1 + (n*6))

        # FINAL BUTTONS #
        self.btn_graph.place(x = 260, y = 375)
        self.btn_exp.place(x = 265, y = 420)

    def selectImportPath(self): # IR
        self.importDir = filedialog.askdirectory()
        path = pathlib.PurePath(self.importDir)
        Label(window, text=path.name, font=10).pack(side = TOP)   

    def selectExportPath(self): # IR
        self.exportDir = filedialog.askdirectory()
        path = pathlib.PurePath(self.exportDir)
        Label(window, text=path.name, font=10).pack(side = BOTTOM)   

    # def enableDependants(self):
        # ungray checkbox bands

    def consolelog(self):
        print("checked event!")

    def graph(self):
        plotScatter = True
        plotBar = False
        control_number = 5
        backend.main(self.importDir, self.exportDir, plotScatter, plotBar, control_number)

window=Tk()
mywin=MyWindow(window, 20, 150, 25, 100)
window.title('Lear Lab IR Integration GUI')
window.geometry("400x450+10+10")
window.mainloop()