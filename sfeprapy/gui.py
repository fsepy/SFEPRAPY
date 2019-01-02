import tkinter as tk


if __name__ == '__main__':
    master = tk.Tk()

    dict_entries = {
        'item 1': int,
        'item 2': str,
    }

    master.title('Hello World!')

    i = 0
    dict_tk_entry = {}
    for key, val in dict_entries.items():
        tk.Label(master, text=str(key)).grid(row=i)
        dict_tk_entry[key] = tk.Entry(master)
        dict_tk_entry[key].grid(row=i, column=1)
        i += 1

    # tk.Label(master, text="First").grid(row=0)
    # tk.Label(master, text="Second").grid(row=1)

    # e1 = tk.Entry(master)
    # e2 = tk.Entry(master)
    #
    # e1.grid(row=0, column=1)
    # e2.grid(row=1, column=1)

    master.mainloop()