import tkinter as tk
#from scripts.Tkinter.tk_commons import OrderBlocks


def run():
    root = tk.Tk()
    oj(root).pack()
    root.mainloop()
    exit(0)


class OrderBlocks(tk.Frame):
    def __init__(self, parent, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)
        self.screen = tk.Frame(self)
        self.screen.pack(fill='both', expand=True)
        self.screen.but = []
        self.screen.sep = []

        self.screen.butlist = [[['A'], ['B', 'C']], [['D']]]
        self.listtocanvas()
        self.screen.selection = tuple()
        self.screen.active = tk.BooleanVar(value=False)
        
    def resetcanvas(self):
        for x in self.screen.but:
            x.destroy()
        for x in self.screen.sep:
            x.destroy()

    def listtocanvas(self):
        self.resetcanvas()

        self.screen.butlist
        for r in self.screen.butlist:
            for j, c in enumerate(r):
                if not c:
                    r.pop(j)

        self.screen.but = []
        self.screen.sep = []
        for i, r in enumerate(self.screen.butlist):
            p = 0
            for j, c in enumerate(r):
                p_ = 0
                for p_, e in enumerate(c):
                    cr, cc = [(i*2)+2, (j*2)+2+(p+p_)]
                    self.screen.but += [tk.Button(
                        self.screen, text=str(e), command=lambda _i=i, _j=j, _p=p_, l=len(self.screen.but): self.sSep(l, _i, _j, _p), height=1, width=3)]
                    self.screen.but[-1].grid(column=cc, row=cr)
                p += p_

        for i, r in enumerate(self.screen.butlist):
            p = 0
            for j, c in enumerate(r):
                p_ = 0
                for p_, e in enumerate(c):
                    cr, cc = [(i*2)+2, (j*2)+2+(p+p_)]
                    for a_ in (0, -1, 1):
                        if not self.screen.grid_slaves(row=cr, column=cc+a_):
                            self.screen.sep += [tk.Button(self.screen, height=1, width=1, bd=0,
                                                          command=lambda _i=i, _j=j+a_: self.sBut(_i, _j))]
                            self.screen.sep[-1].grid(column=cc+a_, row=cr)
                    for a_ in (0, -1, 1):
                        if not self.screen.grid_slaves(row=cr+a_, column=cc):
                            self.screen.sep += [tk.Button(self.screen, height=1, width=1, bd=0,
                                                          command=lambda _i=i+a_, _j=j: self.sBut(_i, _j))]
                            self.screen.sep[-1].grid(column=cc, row=cr+a_)
                p += p_
    
    def canvastolist(self):
        self.screen.butlist = []

        for r in range(len(self.screen.but) + len(self.screen.sep)):
            l_ = [[]]
            for c in self.screen.grid_slaves(row=r):
                if str(c).rsplit('!', 1)[-1].startswith('button') and c.cget('text'):
                    l_[-1] += [c.cget('text')]
                elif l_[-1]:
                    l_ += [[]]
            if l_ != [[]]:
                self.screen.butlist += [l_]

    def sSep(self, i, row, col, item):
        if self.screen.active.get():
            self.screen.but[i].configure(relief="raised")
            self.screen.active.set(False)
            for x in self.screen.sep:
                x.configure(bg='SystemButtonFace', state='disabled')
            if self.screen.selection != (row, col, item):
                self.sBut(row, col, merge=True)
                self.screen.selection = tuple()
        else:
            self.screen.active.set(True)
            self.screen.but[i].configure(relief="sunken", bg='yellow')
            self.screen.selection = (row, col, item)
            #self.selectionflag.set(True)
            for x in self.screen.but:
                x.configure(bg='SystemButtonFace', state='normal')
            for x in self.screen.sep:
                x.configure(bg='red', state='normal')

    def sBut(self, nrow, ncol, merge=False):
        orow, ocol, oite = self.screen.selection
        
        if nrow == -1:
            self.screen.butlist = [[]] + self.screen.butlist
            nrow = 0
            orow += 1
        if len(self.screen.butlist) <= nrow:
            self.screen.butlist += [[] * (1+abs(nrow)-len(self.screen.butlist))]
        
        if len(self.screen.butlist[nrow]) <= ncol:
            self.screen.butlist[nrow] += [] * (1+abs(ncol)-len(self.screen.butlist[nrow]))
        if ncol == -1:
            #self.screen.butlist[nrow] = [[]] + self.screen.butlist[nrow]
            ncol = 0
        
        if merge:
            self.screen.butlist[nrow][ncol] += [self.screen.butlist[orow][ocol].pop(oite)]
        else:
            self.screen.butlist[nrow].insert(
                ncol, [self.screen.butlist[orow][ocol].pop(oite)])
        
        # clean after me
        if not self.screen.butlist[orow][ocol]:
            self.screen.butlist[orow].pop(ocol)
        if not self.screen.butlist[orow]:
            self.screen.butlist.pop(orow)
        print(self.screen.butlist)
        self.listtocanvas()
        for x in self.screen.sep:
            x.configure(bg='SystemButtonFace', state='disabled')
        self.screen.active.set(False)
