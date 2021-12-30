import tkinter

def runGUI():
    window = tkinter.Tk()
    window.columnconfigure(0, weight=1, minsize=75)
    window.rowconfigure(0, weight=1, minsize=50)


    # Code to add widgets will go here...
    greeting = tkinter.Label(text="Enter a stock ticker - downloads last three years of data.")
    greeting.place(x=0, y=0)
    lookup = tkinter.Entry(width=50)
    lookup.place(x=0, y=0)
    search = tkinter.Button(width=10, height=2, text="Download")
    search.place(x=160, y=0)

    frame1 = tkinter.Frame(master=window, width=600, height=100)

    lookup.pack()
    greeting.pack()
    search.pack()
    frame1.pack()




    window.mainloop()