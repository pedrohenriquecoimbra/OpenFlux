#from turtle import update
#from click import option
#import pandas as pd
#import numpy as np

import sys
import tkinter as tk
import PIL
import tkinter.ttk as ttk
import time
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#from tkinter import Tk, Canvas, PhotoImage, mainloop, Button, YES, NO, BOTH, NW, Label, filedialog
#import PIL
#from PIL import Image, ImageDraw, ImageTk
#import matplotlib.pyplot as plt
import subprocess as sp# import Popen, PIPE

def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

class ImageClick():
    def __init__(self, master, image, roi, inventory_item=None, help_text=None):
        self.master = master
        self.roi = roi
        self.image = tk.PhotoImage(file=image)
        self.inventory_item = inventory_item
        self.help_text = help_text
        self.command = print

    def on_click(self, event):
        if (self.roi[0] <= event.x <= self.roi[2]
                and self.roi[1] <= event.y <= self.roi[3]):
            self.command()


class CPython(tk.Frame):
    def __init__(self, parent, *args, _globals=globals(), **kw):
        tk.Frame.__init__(self, parent, *args, **kw)
        self._globals = _globals

        self.cmd = tk.Text(self, height=1, bg='blue', fg='white', font=('Consolas', 12))
        self.cmd.pack(side=tk.BOTTOM, fill=tk.X)#, expand=True)
        self.shw = tk.StringVar(value=f'Python {sys.version}\n')
        #self.prt = tk.Label(self.prompt.interior, textvariable=self.shw, anchor=tk.W, bg='black', fg='white', font=('Consolas', 12))
        self.prt = tk.Text(self, height=1, state='normal', bg='black', fg='white', font=('Consolas', 12))
        self.prt.insert(tk.INSERT, self.shw.get())
        self.prt.config(state="disabled")
        self.ysc = tk.Scrollbar(self, command=self.prt.yview)
        self.prt['yscrollcommand'] = self.ysc.set
        self.ysc.pack(side=tk.RIGHT, fill=tk.Y)
        self.prt.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.cmd.bind('<Return>', self.run)

    def run(self, event):
        command = self.cmd.get('insert linestart', 'insert lineend')
        self.shw.set(self.shw.get() + '> ' + command + '\n')
        
        if command == 'exit':
            print('exit')
            # exit()
        p = eval(command, self._globals)
        self.shw.set('{}{}\n'.format(self.shw.get(), str(p)))
        self.prt.config(state="normal")
        self.prt.insert(tk.INSERT, self.shw.get())
        self.prt.config(state="disabled")
        self.prt.yview_moveto('1.0')

class Prompt(tk.Frame):
    def __init__(self, parent, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)

        self.cmd = tk.Text(self, height=1, bg='blue', fg='white', font=('Consolas', 12))
        self.cmd.pack(side=tk.BOTTOM, fill=tk.X)#, expand=True)
        self.shw = tk.StringVar()
        #self.prt = tk.Label(self.prompt.interior, textvariable=self.shw, anchor=tk.W, bg='black', fg='white', font=('Consolas', 12))
        self.prt = tk.Text(self, height=1, state='disabled', bg='black', fg='white', font=('Consolas', 12))
        self.ysc = tk.Scrollbar(self, command=self.prt.yview)
        self.prt['yscrollcommand'] = self.ysc.set
        self.ysc.pack(side=tk.RIGHT, fill=tk.Y)
        self.prt.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.cmd.bind('<Return>', self.run)

    def com(self, command):
        self.shw.set(self.shw.get() + '> ' + ''.join(list(command)) + '\n')
        if command == 'exit':
            exit()
        p = sp.Popen(command, shell=True, bufsize=10,
            stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, close_fds=True)
        m, e = p.communicate()
        #self.prt.insert('end', f'\n{m}')
        self.shw.set('{}{}\n'.format(self.shw.get(), str(m)))#.decode(("utf-8"))))

    def run(self, event):
        command = self.cmd.get('insert linestart', 'insert lineend')
        self.shw.set(self.shw.get() + '> ' + command + '\n')
        
        if command == 'exit':
            print('exit')
            # exit()
        p = sp.Popen(command, shell=True, bufsize=10,
            stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, close_fds=False)
        m, e = p.communicate()
        #self.prt.insert('end', f'\n{m}')
        self.shw.set('{}{}{}\n'.format(self.shw.get(), m.decode(("utf-8")), e.decode(("utf-8"))))
        self.prt.config(state="normal")
        self.prt.insert(tk.INSERT, self.shw.get())
        self.prt.config(state="disabled")
        self.prt.yview_moveto('1.0')


class VertNotebook(ttk.Frame):
    # https://stackoverflow.com/questions/60750354/is-there-a-way-to-use-a-vertical-notebook-that-has-scrollable-tabs-on-tkinter
    def __init__(self, *args, **kw):
        ttk.Frame.__init__(self, *args, **kw)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(2, weight=1)
        # scrollable tabs
        self._listbox = tk.Listbox(self, width=1, background='lightgrey',
                                   highlightthickness=0, relief='raised')
        scroll = ttk.Scrollbar(self, orient='vertical',
                               command=self._listbox.yview)
        self._listbox.configure(yscrollcommand=scroll.set)

        # list of widgets associated with the tabs
        self._tabs = []
        self._current_tab = None  # currently displayed tab

        scroll.grid(row=0, column=0, sticky='ns')
        self._listbox.grid(row=0, column=1, sticky='ns')
        # binding to display the selected tab
        self._listbox.bind('<<ListboxSelect>>', self.show_tab)

    def add(self, widget, label):  # add tab
        self._listbox.insert('end', label)  # add label listbox
        # resize listbox to be large enough to show all tab labels
        self._listbox.configure(
            width=max(self._listbox.cget('width'), len(label)))
        if self._current_tab is not None:
            self._current_tab.grid_remove()
        self._tabs.append(widget)
        widget.grid(in_=self, column=2, row=0, sticky='ewns')
        self._current_tab = widget
        self._listbox.selection_clear(0, 'end')
        self._listbox.selection_set('end')
        self._listbox.see('end')

    def show_tab(self, event):
        print(event, self._listbox.curselection(), )
        try:
            widget = self._tabs[self._listbox.curselection()[0]]
            print(widget)
        except IndexError:
            return
        if self._current_tab is not None:
            self._current_tab.grid_remove()
        self._current_tab = widget
        widget.grid(in_=self, column=2, row=0, sticky='ewns')


class OrderBlocks(tk.Frame):
    def __init__(self, parent, *args, list=[[['A'], ['B', 'C']], [['D']]],
                 garbage=False, edit=True, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)
        self.editable = tk.BooleanVar(value=True if edit == True else False)
        self.editable.trace_add("write", lambda *a: self.listtocanvas())
        
        self.root = tk.Frame(self)
        self.root.pack(fill='both', expand=True)
        self.garbage = tk.Label(self.root)
        self.garbage.pack(fill="x", expand=True, side="bottom", anchor="s")
        self.unused = tk.Frame(self.root)
        if garbage:
            self.unused.pack(fill='y', expand=False, side="left")
        self.unused.but = []
        self.screen = tk.Frame(self.root)
        self.screen.pack(fill='both', expand=True, side="right")
        self.screen.but = []
        self.screen.sep = []
        self.screen.its = {}
        #self.screen.bfr = []
        #self.screen.chk = []

        self.screen.butlist = list#[[['A'], ['B', 'C']], [['D']]]
        self.unused.butlist = []#["E", "F", "G"]
        self.listtocanvas()
        self.screen.selection = tuple()
        self.screen.active = tk.BooleanVar(value=False)
        
    def resetcanvas(self):
        for x in self.unused.but + \
            [e["frame"] for e in self.screen.its.values() if isinstance(e, dict) and "frame" in e.keys()] + \
                self.screen.sep:
            x.destroy()

    def addbuttom(self, e):
        if e in list(flatten(self.screen.butlist)):
            tk.messagebox.showerror(message="Button with the same name already exists.")
            return
        self.unused.butlist = [x for x in self.unused.butlist if x != e]#.pop(i)
        self.screen.butlist += [[[e]]]
        self.listtocanvas()

    def drpbuttom(self, e, **kw):
        self.screen.butlist = [[[xxx for xxx in xx if xxx != e] for xx in x] for x in self.screen.butlist]
        self.sSep(**kw)
        self.listtocanvas()

    def unsbuttom(self, e, **kw):
        self.unused.butlist += [e]
        self.drpbuttom(e, **kw)

    def returnselected(self):
        selected = [k for k, v in self.screen.its.items() if v["check"].get()]
        selected = [[[i for i in c] for j, c in enumerate(r) if f"{i}_{j}" in selected] for i, r in enumerate(self.screen.butlist)]
        selected = [[c for c in r if c] for r in selected if r]
        return selected

    def listtocanvas(self):
        self.resetcanvas()

        for r in self.screen.butlist:
            for j, c in enumerate(r):
                if not c:
                    r.pop(j)

        self.screen.but = []
        self.screen.sep = []
        self.screen.its = {}
        #self.screen.bfr = []
        #self.screen.chk = []

        self.unused.butlist = [c for c in self.unused.butlist if c]    
        for i, r in enumerate([b for b in self.unused.butlist]):
            self.unused.but += [tk.Button(self.unused, text=str(r), command=lambda i=i, e=r: self.addbuttom(i, e), height=1)]
            self.unused.but[-1].pack()
        
        for i, r in enumerate(self.screen.butlist):
            p = 0
            for j, c in enumerate(r):
                cr, cc = [(i*2)+2, (j*2)+2]
                its = {"frame": tk.Frame(self.screen, borderwidth=1, bg="white"),
                       "check": tk.IntVar(value=1), "cover": tk.Label()}
                
                #self.screen.bfr += [tk.Frame(self.screen, borderwidth=1, bg="white")]
                its["frame"].grid(column=cc, row=cr, stick='w')
                #self.screen.its[f'{i}_{j}'] = {"frame": tk.Frame(self.screen, borderwidth=1, bg="white")}
                #self.screen.chk += [tk.IntVar(value=1)]
                tk.Checkbutton(its["frame"], variable=its["check"]).grid(column=0, row=0)
                p_ = 0
                for p_, e in enumerate(c):
                    #cr, cc = [(i*2)+2, (j*2)+2+(p+p_)]
                    self.screen.but += [tk.Button(
                        its["frame"], text=str(e), command=lambda _i=i, _j=j, _p=p_, l=len(self.screen.but): self.sSep(l, _i, _j, _p), height=1)]
                    self.screen.but[-1].__frame__ = f'{i}_{j}'
                    self.screen.but[-1].req = [len(self.screen.but)-1, i, j, p_]
                    self.screen.but[-1].grid(column=p_+1, row=0)
                p += p_
                its['addbut'] = tk.Button(
                    its["frame"], text=" "*6, command=lambda _i=i, _j=j: self.sBut(_i, _j, merge=True), height=1)
                self.screen.its[f'{i}_{j}'] = its
                del its

        if self.editable.get():
            self.screen.configure(bg='SystemButtonFace')
            for i, r in enumerate(self.screen.butlist):
                for j, c in enumerate(r):
                    cr, cc = [(i*2)+2, (j*2)+2]
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
        else:
            self.screen.configure(bg='SystemButtonFace')
    
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

    def hFra(self, row, col):
        # if click on top of frame
        self.sBut(row, col, merge=True)
        self.screen.selection = tuple()

    def sSep(self, i, row, col, item):
        if not self.editable.get() or self.screen.active.get():
            self.screen.but[i].configure(relief="raised")
            self.screen.active.set(False)
            for x in self.screen.sep:
                x.configure(bg='SystemButtonFace', state='disabled')
            self.unused.configure(bg='SystemButtonFace')
            self.unused.unbind("<Button-1>")
            self.garbage.configure(bg='SystemButtonFace', text="")
            self.garbage.unbind("<Button-1>")
            for k, v in self.screen.its.items():
                if k != self.screen.but[i].__frame__:
                    v['frame'].unbind("<Enter>")
                    v['frame'].unbind("<Leave>")
                    v['addbut'].unbind("<Enter>")
                    v['addbut'].unbind("<Leave>")
                    v['addbut'].grid_forget()
                    #v['cover'].destroy()
                    #v['cover'] = tk.Label()
            
            if not self.editable.get():
                return item

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
                x.configure(bg='gray', state='normal')
                x.bind("<Enter>", lambda *a, x_=x: x_.configure(bg='red'))
                x.bind("<Leave>", lambda *a, x_=x: x_.configure(bg='gray'))
            # add to each frame a label the same size to click on it
            #self.screen.but[i].configure(command=lambda *a, x_=k: self.sBut(*x_.split('_'), merge=True))
            
            for k, v in self.screen.its.items():
                if k != self.screen.but[i].__frame__:
                    print(v["frame"].winfo_width())
                    #v['cover'] = tk.Label(v["frame"], width=10, borderwidth=1)
                    #v['cover'].place(relx=0, rely=0)
                    #v['cover'].bind('<Button-1>', lambda *a: print("move here"))
                    v['frame'].bind("<Enter>", lambda *a, x_=v['frame']: x_.configure(bg='red'))
                    v['frame'].bind("<Leave>", lambda *a, x_=v['frame']: x_.configure(bg='gray'))
                    v['addbut'].bind("<Enter>", lambda *a, x_=v['addbut']: x_.configure(bg='red'))
                    v['addbut'].bind("<Leave>", lambda *a, x_=v['addbut']: x_.configure(bg='SystemButtonFace'))
                    v['addbut'].grid(row=0)
                    #v['frame'].bind("<Button-1>", lambda *a, x_=k: self.sBut(*x_.split('_'), merge=True))
            
            self.unused.configure(bg='red')
            self.unused.bind("<Button-1>", lambda *a, e=self.screen.but[i].cget('text'), 
                             kw={"i": i, "row": row, "col": col, "item": item}: self.unsbuttom(e=e, **kw))
            self.garbage.configure(bg='gray', text="trash")
            self.garbage.bind("<Button-1>", lambda *a, e=self.screen.but[i].cget('text'), 
                              kw={"i": i, "row": row, "col": col, "item": item}: self.drpbuttom(e=e, **kw))

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
        
        self.listtocanvas()
        for x in self.screen.sep:
            x.configure(bg='SystemButtonFace', state='disabled')
        self.screen.active.set(False)


class OrderBlocksMove(tk.Frame):
    def __init__(self, parent, blist, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)
        self.blk_dic = {i: b for i, b in enumerate(blist)}

        self.screen = tk.Frame(self)
        self.screen.pack(fill='both', expand=True)
        self.screen.s = tk.Label(self.screen, height=0)
        self.create_buttons()

    def create_buttons(self):
        self.ChoiceUsrDev = tk.IntVar()
        self.screen.but = [tk.Button(
            self.screen, text=b, command=lambda x=i: self.sBut(x), height=1, width=5) for i, b in self.blk_dic.items()]
        [b.grid(column=1, row=i+1) for i, b in enumerate(self.screen.but)]

    def get(self):
        max_lim = 1000
        y = 0
        order = []
        while y < max_lim:
            # do not preserve order in row
            b = [r.cget('text') for r in self.screen.grid_slaves(row=y)
                 if str(r).rsplit('!', 1)[-1].startswith('button')]
            if b:
                order += [b]
            y += 1
        return order

    def sBut(self, b):
        for i, but in enumerate(self.screen.but):
            if i == b:
                but.bind("<B1-Motion>", lambda e, i=b: self.bMove(b=i))
                but.configure(bg='red')
            else:
                but.unbind("<B1-Motion>")
                but.configure(bg='white')
        return

    def bMove(self, b):
        wid_size = (self.screen.but[b].winfo_width(),
                    self.screen.but[b].winfo_height())
        wid_ss = (1, 5)
        x_ = int(round((self.screen.winfo_pointerx() -
                 self.screen.winfo_rootx()) / wid_size[0]))
        y_ = int(round((self.screen.winfo_pointery() -
                 self.screen.winfo_rooty()) / wid_size[1]))
        #print(x_, y_, wid_ss[1], wid_ss[0], self.screen.grid_slaves(row=y_, column=x_)[
        #      0] if self.screen.grid_slaves(row=y_, column=x_) else '')
        #print(self.screen.grid_slaves(row=y_))

        if (x_ > 0) * (y_ > 0):
            for x in range(1, x_+5):
                if x < x_:
                    if self.screen.grid_slaves(column=x) == []:
                        tk.Label(self.screen, height=wid_ss[1], width=wid_ss[0], text='').grid(
                            row=y_, column=x)
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

class VerticalScrolledFrame(ttk.Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame.
    * Construct and pack/place/grid normally.
    * This frame only allows vertical scrolling.
    * Based on
    * https://web.archive.org/web/20170514022131id_/http://tkinter.unpythonic.net/wiki/VerticalScrolledFrame
    """

    def __init__(self, parent, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)

        # Create a canvas object and a vertical scrollbar for scrolling it.
        vscrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        vscrollbar.pack(fill=tk.Y, side=tk.RIGHT, expand=tk.FALSE)
        self.canvas = canvas = tk.Canvas(self, bd=0, highlightthickness=0,
                                         yscrollcommand=vscrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
        vscrollbar.config(command=canvas.yview)

        # Reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # Create a frame inside the canvas which will be scrolled with it.
        self.interior = interior = ttk.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=tk.NW)

        # Track changes to the canvas and frame width and sync them,
        # also updating the scrollbar.
        def _configure_interior(event):
            # Update the scrollbars to match the size of the inner frame.
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # Update the canvas's width to fit the inner frame.
                canvas.config(width=interior.winfo_reqwidth())
            if interior.winfo_reqheight() != canvas.winfo_height():
                # Update the canvas's height to fit the inner frame.
                canvas.config(height=interior.winfo_reqheight())
        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # Update the inner frame's width to fill the canvas.
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
            """
            if interior.winfo_reqheight() != canvas.winfo_height():
                # Update the inner frame's height to fit the canvas.
                canvas.itemconfigure(interior_id, height=canvas.winfo_height())
            """
        canvas.bind('<Configure>', _configure_canvas)


class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


class CanvasPaint(tk.Canvas):
    def __init__(self, parent, bg=None, *args, **kw):
        self.lastx, self.lasty = None, None
        self.image_number = 0
        self.time = time.time()
        self.w = []

        if isinstance(parent, tk.Canvas):
            tk.Canvas.__init__(self, parent, *args, **kw)
            self.canvas = parent
        else:
            tk.Canvas.__init__(self, parent)#, *args, **kw)

            self.canvas = tk.Canvas(parent)
            #bgimg = tk.PhotoImage('data/pub/static/icos_pub_pauline.jpg')
            #cv = Label(root, i=bgimg)

        # --- PIL
        self.image = bg if bg != None else PIL.Image.new('RGB', self.canvas.size())
        self.draw = PIL.ImageDraw.Draw(self.image)

        #self.canvas.bind('<Button-1>', self.update_lastloc)
        self.canvas.bind('<Motion>', self.pollock)
        self.canvas.pack(expand=True, fill="both")
        return

    def save(self):
        # image_number increments by 1 at every save
        filename = f'data/temp/drawing_{self.image_number}.png'

        # image_number increments by 1 at every save
        # filename = f'data/temp/drawing_{image_number}.png'
        filename = tk.filedialog.asksaveasfilename(initialdir="data/pub/",
                                                   title="Select a File",
                                                   filetypes=((".png", "*.png*"),
                                                              ("all files", "*.*")))
        self.image.save(filename)
        self.image_number += 1

    def update_lastloc(self, e):
        self.time = time.time()
        self.lastx, self.lasty = e.x, e.y

    def pollock(self, e):
        w = min(1+min(3+int((time.time()-self.time)*30), 30)+int((time.time()-self.time)*10), 100)
        self.w = self.w[-100:] + [w]
        self.paint(e, w=int(max(min(max(self.w)/5, 3), 1)))
        self.canvas.create_oval((e.x-w, e.y-w, e.x+w, e.y+w), fill='black')
        #  --- PIL
        self.draw.ellipse((e.x-w, e.y-w, e.x+w, e.y+w), fill='black')

    def paint(self, e, w=1):
        if self.lastx is None or self.lasty is None:
            self.update_lastloc(e)
        x, y = e.x, e.y
        self.canvas.create_line((self.lastx, self.lasty, x, y), width=w)
        #  --- PIL
        self.draw.line((self.lastx, self.lasty, x, y), fill='black', width=w)
        self.update_lastloc(e)


def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)

    def enter(event):
        toolTip.showtip(text)

    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)
