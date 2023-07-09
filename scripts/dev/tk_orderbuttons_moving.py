import tkinter as tk
#from scripts.Tkinter.tk_commons import OrderBlocks


def run():
    root = tk.Tk()
    oj(root).pack()
    root.mainloop()
    exit(0)


class oj(tk.Frame):
    def __init__(self, parent, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)
        self.screen = tk.Frame(self)
        self.screen.pack(fill='both', expand=True)
        self.screen.s = tk.Label(self.screen, height=0)
        self.create_buttons()

    def get(self):
        max_lim = 1000
        y = 0
        order = []
        while y < max_lim:
            # do not preserve order in row
            b = [r.cget('text') for r in self.screen.grid_slaves(row=y)
                     if str(r).rsplit('!', 1)[-1].startswith('button')]
            if b: order += [b]
            y += 1
        return order
        
    def create_buttons(self):
        self.ChoiceUsrDev = tk.IntVar()
        #tk.Radiobutton(self.FMenuRadio, text='User', indicatoron=False, variable=self.ChoiceUsrDev, value=0, command=self.dev_options, width=8).pack(side="left")
        #tk.Radiobutton(self.FMenuRadio, text='Developper', indicatoron=False, variable=self.ChoiceUsrDev, value=1, command=self.dev_options, width=8).pack(side="left")
                                                                                                                                                           
        #self.screen.but = [tk.Radiobutton(
        #    self.screen, text=str(i), variable=self.ChoiceUsrDev, value=i, height=2, width=3) for i in range(5)]
        self.screen.but = [tk.Button(
            self.screen, text=str(i), command=lambda b=i: self.sBut(b), height=1, width=3) for i in range(5)]
        #self.screen.but[0].place(x=400, y=400)
        [self.screen.but[i].grid(column=1, row=i+1) for i in range(5)]
        
    def sBut(self, b):
        print(self.get())
        for i, but in enumerate(self.screen.but):
            if i == b: 
                but.bind("<B1-Motion>", lambda e, i=b: self.bMove(b=i))
                but.configure(bg='red')
            else:
                but.unbind("<B1-Motion>")
                but.configure(bg='white')
        return

    def bMove(self, b):
        #print(e.x, e.y, e, self.screen.winfo_rootx(), self.screen.winfo_rooty())
        wid_size = (self.screen.but[b].winfo_width(),
                    self.screen.but[b].winfo_height())
        wid_ss = (1, 3)
        x_ = int(round((self.screen.winfo_pointerx() - self.screen.winfo_rootx()) / wid_size[0]))
        y_ = int(round((self.screen.winfo_pointery() - self.screen.winfo_rooty()) / wid_size[1]))
        print(x_, y_, wid_ss[1], wid_ss[0], self.screen.grid_slaves(row=y_, column=x_)[0] if self.screen.grid_slaves(row=y_, column=x_) else '')
        #print(self.screen.grid_slaves(row=y_))

        #xy_old = self.screen.but[b].gri
        if (x_ > 0) * (y_ > 0): 
            for x in range(1, x_+5):
                if x < x_:
                    if self.screen.grid_slaves(column=x) == []:
                        tk.Label(self.screen, height=wid_ss[1], width=wid_ss[0], text='').grid(row=y_, column=x)
                if x > x_:
                    
                    [r.destroy() for r in self.screen.grid_slaves(column=x)
                         if str(r).rsplit('!', 1)[-1].startswith('label')]
                    
            for y in range(1, y_+5):
                if y < y_:
                    if self.screen.grid_slaves(row=y) == []:
                        tk.Label(self.screen, height=wid_ss[1], width=wid_ss[0]).grid(
                            row=y, column=x_)
                if y > y_:
                    [c.destroy() for c in self.screen.grid_slaves(row=y)
                         if str(c).rsplit('!', 1)[-1].startswith('label')]
            
            if self.screen.grid_slaves(row=y_, column=x_) and str(self.screen.grid_slaves(row=y_, column=x_)[0]).rsplit('!', 1)[-1].startswith('label'):
                self.screen.grid_slaves(row=y_, column=x_)[0].destroy()
                
            self.screen.but[b].grid(row=y_, column=x_)
            self.screen.update()
        #self.screen.but[self.ChoiceUsrDev.get()].place(x=e.x, y=e.y)

        #self.screen.focus_get().place(x=e.x, y=e.y)


run()
