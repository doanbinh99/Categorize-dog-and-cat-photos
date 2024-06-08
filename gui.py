import tkinter as tk
from tkinter import *
from tkinter import messagebox, filedialog
from PIL import ImageTk, Image
import numpy
from keras.models import load_model

model = load_model('Model.h5')
classes = {
    0: 'đây là con mèo!',
    1: 'đây là con chó!',
}


def upload_image_canvas(frame):
    for widget in frame.winfo_children():
        widget.destroy()
    canvas = Canvas(frame, width=820, height=400)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)

    v_scrollbar = Scrollbar(frame, orient="vertical", command=canvas.yview)
    v_scrollbar.pack(side=RIGHT, fill=Y)
    canvas.configure(yscrollcommand=v_scrollbar.set)

    # Tạo frame con trong canvas
    canvas_frame = Frame(canvas)
    # Tạo cửa sổ trong canvas, có canvas_frame làm nội dung
    canvas.create_window((0, 0), window=canvas_frame, anchor="nw")

    canvas_frame.grid_rowconfigure(0, weight=1)
    canvas_frame.grid_columnconfigure(0, weight=1)

    browser_image(canvas_frame)

    canvas_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))


def browser_image(frame):
    try:
        global file_paths
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png *.gif *.jpeg")])
        i = 0
        j = 0
        for file_path in file_paths:
            image = Image.open(file_path)
            image = image.resize((150, 150))
            # image.thumbnail((int(window.winfo_width()/2.25), int(window.winfo_height()/2.25)))
            img = ImageTk.PhotoImage(image)
            upload_image = Label(frame, image=img)
            upload_image.image = img
            upload_image.grid(row=i, column=j, padx=5, pady=5)
            j += 1
            if j > 4:
                i += 1
                j = 0
        frame_content.configure(text=f"Ảnh tải lên ({len(file_paths)})")
    except (Exception, ):
        messagebox.showerror('Error', "Vui lòng chọn lại ảnh !")


def recognize_dog_cat(file_path):
    image = Image.open(file_path)
    image = image.resize((128, 128))
    image = numpy.expand_dims(image, axis=0)  # biến ảnh đầu vào thảnh một mảng có kích thước (1, 128, 128, 3)
    image = numpy.array(image)  # chuyển đổi đối tượng iamge thành một mảng
    image = image / 255  # chuẩn hóa giá trị điểm ảnh về khoảng [0, 1]
    pred = model.predict(image)
    index = numpy.argmax(pred)  # trả về chỉ số của phần tử lớn nhất trong mảng(đại diện cho nhãn được dự đoán)
    return index


def classify():
    try:
        dogs.clear()
        cats.clear()
        for file_path in file_paths:
            index = recognize_dog_cat(file_path)
            if index == 0:
                cats.append(file_path)
                print("Cat")
            else:
                dogs.append(file_path)
                print("Dog")

        child = child_window(window)

        frame_cat = LabelFrame(child, text='', background='#ccc', font=('arial', 10))
        frame_cat.grid(row=1, column=0, padx=10)
        output_frame(frame_cat, cats, "Mèo")

        frame_dog = LabelFrame(child, text='', background='#ccc', font=('arial', 10))
        frame_dog.grid(row=1, column=1, padx=10)
        output_frame(frame_dog, dogs, "Chó")
    except (Exception, TypeError):
        messagebox.showerror("Error", "Vui lòng tải ảnh lên !")


def child_window(parent):
    # Tạo một đối tượng cửa sổ con có cha là parent
    child = Toplevel(parent)
    # Đặt kích thước và tiêu đề của cửa sổ con
    child.geometry('1090x600')
    child.title("Kết quả phân loại")
    child.configure(background='#ccc')
    Label(child, text="Kết quả phân loại",
          pady=20, font=('arial', 20, 'bold'),
          background='#ccc', foreground='#090852').grid(row=0, column=0, columnspan=2, sticky='news', pady=20)
    frame_btn = Frame(child, background='#ccc')
    frame_btn.place(relx=0, rely=0)
    Button(frame_btn, text="Lưu kết quả", background='#28217a', foreground='white',
           font=('arial', 10, 'bold'), command=save_image, padx=10, pady=5).pack(pady=5, padx=5)
    Button(frame_btn, text="Thoát", background='#28217a', foreground='white',
           font=('arial', 10, 'bold'), command=exit, padx=10, pady=5).pack(pady=5)
    return child


def output_frame(frame, lst, name):
    for widget in frame.winfo_children():
        widget.destroy()
    canvas = Canvas(frame, width=500, height=400)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)

    v_scrollbar = Scrollbar(frame, orient="vertical", command=canvas.yview)
    v_scrollbar.pack(side=RIGHT, fill=Y)
    canvas.configure(yscrollcommand=v_scrollbar.set)

    # Tạo frame con trong canvas
    canvas_frame = Frame(canvas)
    # Tạo cửa sổ trong canvas, có canvas_frame làm nội dung
    canvas.create_window((0, 0), window=canvas_frame, anchor="nw")

    canvas_frame.grid_rowconfigure(0, weight=1)
    canvas_frame.grid_columnconfigure(0, weight=1)

    i = 0
    j = 0
    for file_path in lst:
        image = Image.open(file_path)
        image = image.resize((150, 150))
        img = ImageTk.PhotoImage(image)
        upload_image = Label(canvas_frame, image=img)
        upload_image.image = img
        upload_image.grid(row=i, column=j, padx=5, pady=5)
        j += 1
        if j > 2:
            i += 1
            j = 0
    frame.configure(text=f"Ảnh {name} ({len(lst)})")
    canvas_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))


def save_image():
    for i in range(len(dogs)):
        image = Image.open(dogs[i])
        image.save(f"./result/dogs/dog_{i}.jpg")
    for i in range(len(cats)):
        image = Image.open(cats[i])
        image.save(f"./result/cats/cat_{i}.jpg")
    messagebox.showinfo("Thông báo", "Lưu kết quả thành công")


def create_btn_menu():
    btn_upload = Button(frame_menu, text="Tải ảnh lên", background='#28217a', foreground='white',
                        font=('arial', 10, 'bold'), command=lambda: upload_image_canvas(frame_content), padx=10, pady=5)
    btn_upload.pack(fill=BOTH, padx=5, pady=10)

    btn_classify = Button(frame_menu, text="Phân loại", background='#28217a', foreground='white',
                          font=('arial', 10, 'bold'), command=classify, padx=10, pady=5)
    btn_classify.pack(fill=BOTH, padx=5, pady=10)


window = tk.Tk()
window.geometry('1000x600')
window.resizable(False, False)
window.title("Phân loại chó mèo")
window.configure(background='#CCC')

dogs = []
cats = []
Label(window, text="Phân loại chó mèo",
      pady=20, font=('arial', 20, 'bold'),
      background='#ccc', foreground='#090852').place(rely=0.02, relx=0.4)

frame_menu = LabelFrame(window, text="Menu", background='#ccc', font=('arial', 10))
frame_menu.place(relx=0.015, rely=0.15)
create_btn_menu()

frame_content = LabelFrame(window, text="Ảnh tải lên", background='#ccc', font=('arial', 10))
frame_content.place(relx=0.14, rely=0.15)

window.mainloop()
