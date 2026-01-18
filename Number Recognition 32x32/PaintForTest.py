import tkinter as tk
import numpy as np
import tensorflow as tf

miuX = np.load(".\\data\\miuX.npy")
sigmaX = np.load(".\\data\\sigmaX.npy")
model = tf.keras.models.load_model(".\\data\\28x28paint.keras")

PIXEL_SIZE = 20 
GRID_SIZE = 28

img = np.zeros((28, 28), dtype=np.float32)

def paint(event):
    x = event.x // PIXEL_SIZE
    y = event.y // PIXEL_SIZE

    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
        img[y, x] = 255
        canvas.create_rectangle(
            x * PIXEL_SIZE,
            y * PIXEL_SIZE,
            (x + 1) * PIXEL_SIZE,
            (y + 1) * PIXEL_SIZE,
            fill="white",
            outline=""
        )

def clear():
    global img
    img.fill(0)
    canvas.delete("all")
    result_label.config(text="Predicción: ")
    
def predict():
    x = img.reshape(1, 784)


    x = (x - miuX) / sigmaX

    preds = model.predict(x, verbose=0)
    digit = np.argmax(preds)

    result_label.config(text=f"Predicción: {digit}")

root = tk.Tk()
root.title("MNIST Paint")

canvas = tk.Canvas(
    root,
    width=GRID_SIZE * PIXEL_SIZE,
    height=GRID_SIZE * PIXEL_SIZE,
    bg="black"
)
canvas.pack()
canvas.bind("<B1-Motion>", paint)

btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Predict", command=predict).pack(side="left")
tk.Button(btn_frame, text="Clear", command=clear).pack(side="left")

result_label = tk.Label(root, text="Predicción: ")
result_label.pack()

root.mainloop()