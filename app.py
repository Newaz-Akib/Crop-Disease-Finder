from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = ['Disease_A', 'Healthy']
model = tf.keras.models.load_model(os.path.join("models", "model.h5"))

def preprocess_image(img_path):
    # ট্রেইনিংয়ের সাথে হুবহু ১:১ ম্যাচ করার জন্য প্রিপসেসিং
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    # নোট: রিস্কেলিং (1./255) মডেলের প্রথম লেয়ারেই আছে, তাই এখানে শুধু ডাইমেনশন বাড়ানো হলো
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files: return "No file"
    file = request.files["file"]
    if file.filename == "": return "No file"

    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)

    img_array = preprocess_image(path)
    prediction = model.predict(img_array)[0]
    
    print(f"\n📊 LIVE SCORES -> Disease_A: {prediction[0]:.4f}, Healthy: {prediction[1]:.4f}")

    class_index = np.argmax(prediction)
    result = classes[class_index]
    
    return render_template("index.html", prediction=result, image_path=path.replace('\\', '/'))

if __name__ == "__main__":
    print("\n🚀 --- Server Locked & Running Continuously --- 🚀")
    # debug=True এবংuse_reloader=False সার্ভারকে ক্র্যাশ বা ডিসকানেক্ট হওয়া থেকে পুরোপুরি আটকে রাখবে
    app.run(debug=True, port=5000, use_reloader=False)