import tkinter as tk
from tkinter import messagebox

def change_color():
    selected_color = color_listbox.get(color_listbox.curselection())
    root.configure(bg=selected_color)

root = tk.Tk()
root.title("Color Changer")

color_list = ["red", "green", "blue", "yellow", "orange"]

color_label = tk.Label(root, text="Choose a color:")
color_label.pack()

color_listbox = tk.Listbox(root, selectmode=tk.SINGLE)
for color in color_list:
    color_listbox.insert(tk.END, color)
color_listbox.pack()

change_button = tk.Button(root, text="Change Color", command=change_color)
change_button.pack()

root.mainloop()
